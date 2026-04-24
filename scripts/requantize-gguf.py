#!/usr/bin/env python3
"""Requantize a chatterbox GGUF (T3 or S3Gen) to a smaller dtype.

`llama-quantize` refuses to touch either GGUF because neither
`chatterbox` nor `chatterbox-s3gen` is a llama.cpp-known arch.  This
tool walks the GGUF tensor-by-tensor and rewrites it with the big 2-D
weight matrices stored as `Q8_0` / `Q5_0` / `Q4_0`, leaving the
numerically-sensitive tensors (embedding tables accessed via get_rows,
biases, norm scales, filterbank / STFT bases, positional embeddings,
builtin voice conditioning) at their source dtype.

Works for both models because the deny-list covers the union of
patterns that either side uses for "keep-as-F32/F16".

Usage:

    # T3 Q8_0
    python scripts/requantize-gguf.py \\
        models/chatterbox-t3-turbo.gguf \\
        models/t3-q8_0.gguf q8_0

    # S3Gen Q8_0
    python scripts/requantize-gguf.py \\
        models/chatterbox-s3gen.gguf \\
        models/chatterbox-s3gen-q8_0.gguf q8_0

    # Q4_0 is the same, last arg is just `q4_0`.

Quality trade-off (measured on the QVAC paragraph, Metal / M3 Ultra):
  F32 (default)   — baseline
  Q8_0            — essentially bit-exact, cos-sim > 0.99 vs baseline
  Q4_0            — different CFM ODE trajectory → different sample;
                    subjective quality equal, cos-sim falls to ~0.66
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf


# Names we NEVER touch: they're read as raw F32 by the C++ loader, or
# they're accessed via ggml_get_rows (embedding tables), or they're
# numerically sensitive (filterbanks, STFT bases, voice conditioning,
# position embeddings, norm/bias params).  Works for both T3 (GPT-2-
# style names) and S3Gen (custom per-module names).
_DENY_SUBSTRINGS = (
    # Raw-F32 access in the C++ loader
    "flow/input_embedding",     # S3Gen speech embedding table (read as F32 for CPU-side lookup)
    "/builtin/",                # voice conditioning tensors, loaded directly
    # Embedding tables (accessed via ggml_get_rows — safer as F16/F32)
    "text_emb",                 # T3 text token embedding
    "speech_emb",               # T3 speech token embedding
    "wte",                      # GPT-2 word token embedding
    "wpe",                      # GPT-2 learned position embedding
    # Spectral bases / positional encodings (bit-exact numerics)
    "stft_basis",               # STFT analysis / synthesis
    "mel_filterbank",           # mel filterbank
    "mel_fb",                   # T3 VoiceEncoder and S3Gen mel filterbank tensors
    "pos_emb",                  # positional embeddings — small, keep F32
    "pe/pe",                    # conformer pos enc
    # Biases / norms / scale params — always 1-D or near-1-D
    "/b",                       # legacy biases (gpt-2 /b, s3gen /b)
    "/bias",                    # pytorch-style bias
    "/bn/",                     # batchnorm params
    "/norm/",                   # layernorms
    "/ln_",                     # GPT-2 style layernorms (ln_1, ln_2, ln_f)
    "/g",                       # GPT-2 style norm scale (matches /g, /ga[mma], /gate — accept the occasional false deny)
    "/s",                       # legacy scale weights
    "alpha",                    # Snake activation alphas
    "beta",
    "gamma",
    # Voice-cloning preprocessing encoders — NEVER quantize.  These are
    # small specialised models whose dynamic range is too tight for Q4/Q8
    # block quantization; the resulting encoder output drifts so badly that
    # the voice-cloning tensors become unusable (we've seen speaker_emb
    # collapse to zeros, prompt_token to a single constant value, and
    # CAMPPlus embedding go antipodal to its F32 counterpart).  Keeping
    # them at source dtype costs ~40 MB across both GGUFs but is the
    # difference between a working clone and garbage audio.
    "voice_encoder/",           # T3 VoiceEncoder (3-layer bi-LSTM + projection)
    "campplus/",                # S3Gen CAMPPlus (TDNN x-vector extractor)
    "s3tokv2/",                 # S3Gen S3TokenizerV2 (conformer + FSQ quantizer)
)


# Tensor element dtypes we're willing to quantize from.  F16 is T3's
# default for its big projection weights; F32 is S3Gen's default.
_QUANTIZABLE_SRC_DTYPES = {
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
}


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

        if t.tensor_type in _QUANTIZABLE_SRC_DTYPES and should_quantize(t.name, shape, qtype):
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
