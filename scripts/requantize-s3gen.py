#!/usr/bin/env python3
"""Requantize an existing F32/F16 S3Gen GGUF to a smaller dtype.

The upstream `convert-s3gen-to-gguf.py` emits every tensor as F32 (or
F16 for a few fallbacks).  Running `llama-quantize` on the result
fails because llama.cpp doesn't know the custom `chatterbox-s3gen`
architecture.  This tool walks the GGUF tensor-by-tensor and
rewrites it with the big 2D weight matrices stored as `Q8_0` or
`Q4_0`, leaving the numerically-sensitive tensors (embedding table,
biases, norm scales, filterbank bases, builtin voice conditioning)
at full precision.

Quality trade-off on the QVAC paragraph (Metal, M3 Ultra):
  F32 (default) — baseline
  Q8_0         — essentially bit-exact, <1e-3 cos-diff vs F32 mel
  Q4_0         — audibly identical for prose, minor artefacts on
                 expressive content; ~3x size reduction

Usage:

    python scripts/requantize-s3gen.py \\
        models/chatterbox-s3gen.gguf \\
        models/chatterbox-s3gen-q8_0.gguf \\
        q8_0

    python scripts/requantize-s3gen.py \\
        models/chatterbox-s3gen.gguf \\
        models/chatterbox-s3gen-q4_0.gguf \\
        q4_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf


# Names we NEVER touch: they're read as raw F32 by the C++ loader or
# they're numerically sensitive (filterbanks, STFT bases, voice
# conditioning tensors with strict shape contracts).
_DENY_SUBSTRINGS = (
    "flow/input_embedding",     # read as raw F32 for CPU-side embedding lookup
    "/builtin/",                # voice conditioning, loaded as F32
    "stft_basis",               # STFT analysis / synthesis (bit-exact numerics)
    "mel_filterbank",           # mel filterbank, same story
    "pos_emb",                  # positional embeddings — tiny, keep F32
    "pe/pe",                    # conformer pos enc, tiny
    "/b",                       # legacy biases named "/b"
    "/bias",                    # biases
    "/bn/",                     # batchnorm params
    "/norm/",                   # layernorms
    "/s",                       # legacy scale weights
    "alpha",                    # Snake activation alphas
    "beta",
    "gamma",
)


_QUANT_TYPE = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}


def should_quantize(name: str, shape: tuple[int, ...], qtype: gguf.GGMLQuantizationType) -> bool:
    # Keep tiny tensors at full precision.
    n_elements = 1
    for d in shape:
        n_elements *= d
    if n_elements < 1024:
        return False

    # Deny-list.
    lower = name.lower()
    for s in _DENY_SUBSTRINGS:
        if s in name:  # case-sensitive for path-like names
            return False

    # Quantization needs the reduction dim to be a multiple of the block size.
    # In ggml 2D matmul, weight tensor has shape (ne0, ne1) and ne0 is the
    # reduction dim.  Here GGUFReader exposes shape in numpy (reversed) order,
    # so the reduction dim is shape[-1].
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    if shape[-1] % block != 0:
        return False

    # Stick to 2D (plain matmul) and 3D (conv with kernel_size as leading dim).
    # Convs can be quantized in ggml since im2col produces F32 data which
    # mul_mat handles against Q-weights; but we play it safe and only
    # quantize the 2D matmul weights where we know ggml_mul_mat is used.
    if len(shape) != 2:
        return False

    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="Source GGUF (F32/F16)")
    ap.add_argument("dst", type=Path, help="Output GGUF")
    ap.add_argument("dtype", choices=_QUANT_TYPE.keys(), help="Target quant dtype")
    args = ap.parse_args()

    qtype = _QUANT_TYPE[args.dtype]

    src = gguf.GGUFReader(args.src, "r")
    arch = src.fields.get("general.architecture")
    arch_name = ""
    if arch is not None:
        arch_name = bytes(arch.parts[arch.data[0]]).decode("utf-8")

    writer = gguf.GGUFWriter(args.dst, arch_name or "chatterbox-s3gen")

    # Copy all metadata (KV fields) verbatim.  Skip the ones the writer
    # sets itself to avoid duplicates.
    _SKIP_KEYS = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
    }
    for key, field in src.fields.items():
        if key in _SKIP_KEYS:
            continue
        val_type = field.types[0] if field.types else None
        parts = [field.parts[i] for i in field.data]
        if val_type is None:
            continue
        if val_type == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[1] if len(field.types) > 1 else None
            if sub_type == gguf.GGUFValueType.STRING:
                values = [bytes(p).decode("utf-8") for p in parts]
                writer.add_array(key, values)
            else:
                arr = np.concatenate([np.asarray(p) for p in parts]).tolist()
                writer.add_array(key, arr)
        elif val_type == gguf.GGUFValueType.STRING:
            writer.add_string(key, bytes(parts[0]).decode("utf-8"))
        elif val_type == gguf.GGUFValueType.BOOL:
            writer.add_bool(key, bool(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.UINT8, gguf.GGUFValueType.UINT16,
                          gguf.GGUFValueType.UINT32, gguf.GGUFValueType.UINT64):
            writer.add_uint32(key, int(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.INT8, gguf.GGUFValueType.INT16,
                          gguf.GGUFValueType.INT32, gguf.GGUFValueType.INT64):
            writer.add_int32(key, int(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
            writer.add_float32(key, float(parts[0][0]))

    quantized_count = 0
    kept_count = 0
    src_bytes = 0
    dst_bytes = 0

    for t in src.tensors:
        # GGUFReader returns shape in numpy-style reversed order.
        shape = tuple(int(d) for d in reversed(t.shape) if d > 0)
        if not shape:
            shape = (int(t.shape[0]),)

        data = np.asarray(t.data)
        src_bytes += data.nbytes

        if t.tensor_type == gguf.GGMLQuantizationType.F32 and should_quantize(t.name, shape, qtype):
            # Reshape to natural (shape).  GGUF raw data is contiguous in
            # the original order, but reversed() above gives element-shape
            # which is what `quantize()` expects.
            arr = data.astype(np.float32).reshape(shape)
            qdata = gguf.quants.quantize(arr, qtype)
            writer.add_tensor(t.name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
            quantized_count += 1
            dst_bytes += qdata.nbytes
        else:
            # Pass through unchanged.  Preserve original dtype.
            arr = data.reshape(shape)
            writer.add_tensor(t.name, arr, raw_shape=arr.shape, raw_dtype=t.tensor_type)
            kept_count += 1
            dst_bytes += arr.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"arch: {arch_name!r}")
    print(f"quantized: {quantized_count} tensors to {args.dtype.upper()}")
    print(f"kept:      {kept_count} tensors as source dtype")
    print(f"size:      {src_bytes / 1e6:.1f} MB  →  {dst_bytes / 1e6:.1f} MB  "
          f"({dst_bytes / src_bytes * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
