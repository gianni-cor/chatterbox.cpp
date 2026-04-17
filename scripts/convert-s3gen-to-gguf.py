#!/usr/bin/env python3
"""
Convert Chatterbox Turbo S3Gen (flow + mel2wav) weights to GGUF.

Exports:
 - flow.input_embedding            (6561, 512)
 - flow.spk_embed_affine           weight + bias
 - flow.encoder.embed              subsampling layer
 - flow.encoder.pre_lookahead      conv1 + conv2 weights
 - flow.encoder.encoders.{0..5}    6 Conformer blocks (with rel-pos attn)
 - flow.encoder.up_layer           upsample conv
 - flow.encoder.up_embed           second subsampling
 - flow.encoder.up_encoders.{0..3} 4 more Conformer blocks
 - flow.encoder.after_norm         LayerNorm
 - flow.encoder_proj               Linear(512->80)
 - flow.decoder.estimator          ConditionalDecoder (U-Net with transformer blocks)
 - mel2wav.*                       HiFTGenerator (weight_norm convs resolved)

Also embeds built-in S3Gen conditionals:
 - prompt_token  (250,)  int32
 - prompt_feat   (500, 80) float32
 - embedding     (1, 192) float32
"""

import argparse
import re
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


REPO_ID = "ResembleAI/chatterbox-turbo"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("models/chatterbox-s3gen.gguf"))
    ap.add_argument("--hf-token")
    return ap.parse_args()


def as_numpy(tensor: torch.Tensor, *, dtype=None) -> np.ndarray:
    if dtype is not None:
        tensor = tensor.to(dtype)
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


def resolve_weight_norm(state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    """
    PyTorch weight_norm stores original0 (g, magnitudes) and original1 (v, direction).
    Actual weight = g * v / ||v||_2.  For 2D convs we broadcast appropriately.
    Returns the fused weight tensor.
    """
    g = state[f"{prefix}.parametrizations.weight.original0"]
    v = state[f"{prefix}.parametrizations.weight.original1"]
    # ||v|| is computed over all dims except 0 (the output channel dim)
    # by default for Conv1d. See torch.nn.utils.weight_norm.
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.ndim - 1)))
    return g * v / norm


def expand_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Rewrite all `*.parametrizations.weight.original{0,1}` entries into a single
    `*.weight` tensor and drop the originals. Also rename `*.parametrizations.weight.0.original0`
    etc. if present.
    """
    out = dict(state)
    prefixes = set()
    for k in state:
        m = re.match(r"(.+)\.parametrizations\.weight\.original0$", k)
        if m:
            prefixes.add(m.group(1))
    for p in prefixes:
        out[f"{p}.weight"] = resolve_weight_norm(state, p)
        out.pop(f"{p}.parametrizations.weight.original0", None)
        out.pop(f"{p}.parametrizations.weight.original1", None)
    return out


def export(writer: gguf.GGUFWriter, state: dict, name: str, *, dtype=torch.float32):
    arr = as_numpy(state[name], dtype=dtype)
    # Map the name to a GGUF-friendly name but keep the hierarchy recognizable.
    gguf_name = name
    writer.add_tensor(gguf_name, arr)
    return arr.shape


def export_conformer_block(writer: gguf.GGUFWriter, state: dict, prefix: str, gguf_prefix: str):
    """Export one Conformer encoder block."""
    mapping = {
        "norm_mha.weight":           ("norm_mha/w", torch.float32),
        "norm_mha.bias":             ("norm_mha/b", torch.float32),
        "norm_ff.weight":            ("norm_ff/w", torch.float32),
        "norm_ff.bias":              ("norm_ff/b", torch.float32),
        "self_attn.linear_q.weight": ("attn/q/w", torch.float32),
        "self_attn.linear_q.bias":   ("attn/q/b",   torch.float32),
        "self_attn.linear_k.weight": ("attn/k/w", torch.float32),
        "self_attn.linear_k.bias":   ("attn/k/b",   torch.float32),
        "self_attn.linear_v.weight": ("attn/v/w", torch.float32),
        "self_attn.linear_v.bias":   ("attn/v/b",   torch.float32),
        "self_attn.linear_out.weight": ("attn/o/w", torch.float32),
        "self_attn.linear_out.bias":   ("attn/o/b",   torch.float32),
        "self_attn.linear_pos.weight": ("attn/pos/w", torch.float32),
        "self_attn.pos_bias_u":      ("attn/pos_bias_u", torch.float32),
        "self_attn.pos_bias_v":      ("attn/pos_bias_v", torch.float32),
        "feed_forward.w_1.weight":   ("ff/w1/w", torch.float32),
        "feed_forward.w_1.bias":     ("ff/w1/b",   torch.float32),
        "feed_forward.w_2.weight":   ("ff/w2/w", torch.float32),
        "feed_forward.w_2.bias":     ("ff/w2/b",   torch.float32),
    }
    for src_suffix, (dst_suffix, dtype) in mapping.items():
        src = f"{prefix}.{src_suffix}"
        dst = f"{gguf_prefix}/{dst_suffix}"
        arr = as_numpy(state[src], dtype=dtype)
        writer.add_tensor(dst, arr)


def main():
    args = parse_args()
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = Path(snapshot_download(
            repo_id=REPO_ID, token=args.hf_token,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        ))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading s3gen_meanflow from {ckpt_dir}")
    raw = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
    state = expand_weight_norm(raw)

    print(f"Resolved {len([k for k in raw if 'parametrizations' in k])} weight_norm entries")

    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)
    gen = conds["gen"]

    writer = gguf.GGUFWriter(str(args.out), "chatterbox-s3gen")
    writer.add_name("Chatterbox Turbo S3Gen")
    writer.add_description("S3Gen flow + mel2wav (HiFT) for ggml port.")

    # Meta / hparams
    writer.add_uint32("s3gen.speech_vocab_size", 6561)
    writer.add_uint32("s3gen.input_size", 512)
    writer.add_uint32("s3gen.output_size", 80)
    writer.add_uint32("s3gen.encoder.n_blocks", 6)
    writer.add_uint32("s3gen.encoder.up_n_blocks", 4)
    writer.add_uint32("s3gen.encoder.attention_heads", 8)
    writer.add_uint32("s3gen.encoder.head_dim", 64)
    writer.add_uint32("s3gen.encoder.ff_size", 2048)
    writer.add_uint32("s3gen.encoder.token_mel_ratio", 2)
    writer.add_uint32("s3gen.encoder.pre_lookahead_len", 3)
    writer.add_float32("s3gen.layer_norm_eps", 1e-12)
    writer.add_uint32("s3gen.spk_embed_dim", 192)

    # Built-in conditionals
    prompt_token = gen["prompt_token"].reshape(-1).to(torch.int32)
    prompt_feat = gen["prompt_feat"].squeeze(0)          # (500, 80)
    embedding = gen["embedding"].squeeze(0)              # (192,)
    writer.add_uint32("s3gen.builtin.prompt_token_len", int(prompt_token.numel()))
    writer.add_uint32("s3gen.builtin.prompt_feat_frames", int(prompt_feat.shape[0]))
    writer.add_tensor("s3gen/builtin/prompt_token", as_numpy(prompt_token))
    writer.add_tensor("s3gen/builtin/prompt_feat", as_numpy(prompt_feat, dtype=torch.float32))
    writer.add_tensor("s3gen/builtin/embedding", as_numpy(embedding, dtype=torch.float32))

    # Flow top-level weights
    writer.add_tensor("flow/input_embedding",       as_numpy(state["flow.input_embedding.weight"]))
    writer.add_tensor("flow/spk_embed_affine/w",    as_numpy(state["flow.spk_embed_affine_layer.weight"]))
    writer.add_tensor("flow/spk_embed_affine/b",    as_numpy(state["flow.spk_embed_affine_layer.bias"]))
    writer.add_tensor("flow/encoder_proj/w",        as_numpy(state["flow.encoder_proj.weight"]))
    writer.add_tensor("flow/encoder_proj/b",        as_numpy(state["flow.encoder_proj.bias"]))

    # Encoder embed (LinearNoSubsampling: Linear(512 -> 512) + LayerNorm)
    writer.add_tensor("flow/encoder/embed/linear/w",  as_numpy(state["flow.encoder.embed.out.0.weight"]))
    writer.add_tensor("flow/encoder/embed/linear/b",  as_numpy(state["flow.encoder.embed.out.0.bias"]))
    writer.add_tensor("flow/encoder/embed/norm/w",    as_numpy(state["flow.encoder.embed.out.1.weight"]))
    writer.add_tensor("flow/encoder/embed/norm/b",    as_numpy(state["flow.encoder.embed.out.1.bias"]))

    # PreLookaheadLayer: two convs (kernel 4 and 3). Use F32 via custom im2col+matmul.
    writer.add_tensor("flow/encoder/pre_lookahead/conv1/w", as_numpy(state["flow.encoder.pre_lookahead_layer.conv1.weight"]))
    writer.add_tensor("flow/encoder/pre_lookahead/conv1/b", as_numpy(state["flow.encoder.pre_lookahead_layer.conv1.bias"]))
    writer.add_tensor("flow/encoder/pre_lookahead/conv2/w", as_numpy(state["flow.encoder.pre_lookahead_layer.conv2.weight"]))
    writer.add_tensor("flow/encoder/pre_lookahead/conv2/b", as_numpy(state["flow.encoder.pre_lookahead_layer.conv2.bias"]))

    # 6 Conformer blocks
    for i in range(6):
        export_conformer_block(writer, state,
                               f"flow.encoder.encoders.{i}",
                               f"flow/encoder/block{i}")

    # Upsample1D (Conv1d with kernel 5) — F32 (we use conv1d_f32 in C++)
    writer.add_tensor("flow/encoder/up_layer/conv/w", as_numpy(state["flow.encoder.up_layer.conv.weight"]))
    writer.add_tensor("flow/encoder/up_layer/conv/b", as_numpy(state["flow.encoder.up_layer.conv.bias"]))

    # up_embed (second subsampling)
    writer.add_tensor("flow/encoder/up_embed/linear/w", as_numpy(state["flow.encoder.up_embed.out.0.weight"]))
    writer.add_tensor("flow/encoder/up_embed/linear/b", as_numpy(state["flow.encoder.up_embed.out.0.bias"]))
    writer.add_tensor("flow/encoder/up_embed/norm/w",   as_numpy(state["flow.encoder.up_embed.out.1.weight"]))
    writer.add_tensor("flow/encoder/up_embed/norm/b",   as_numpy(state["flow.encoder.up_embed.out.1.bias"]))

    # 4 more Conformer blocks
    for i in range(4):
        export_conformer_block(writer, state,
                               f"flow.encoder.up_encoders.{i}",
                               f"flow/encoder/up_block{i}")

    # Final after_norm
    writer.add_tensor("flow/encoder/after_norm/w", as_numpy(state["flow.encoder.after_norm.weight"]))
    writer.add_tensor("flow/encoder/after_norm/b", as_numpy(state["flow.encoder.after_norm.bias"]))

    # Decoder estimator (CFM) — F32 (we use conv1d_f32 helper).
    decoder_keys = sorted(k for k in state if k.startswith("flow.decoder.estimator."))
    for k in decoder_keys:
        gguf_name = k.replace("flow.decoder.estimator.", "cfm/").replace(".", "/")
        writer.add_tensor(gguf_name, as_numpy(state[k], dtype=torch.float32))

    # mel2wav (HiFTGenerator) — F32 (we use conv1d_f32 helper)
    mel2wav_keys = sorted(k for k in state if k.startswith("mel2wav."))
    for k in mel2wav_keys:
        gguf_name = k.replace("mel2wav.", "hift/").replace(".", "/")
        writer.add_tensor(gguf_name, as_numpy(state[k], dtype=torch.float32))

    n_flow = sum(1 for k in state if k.startswith("flow.")) - sum(1 for k in state if k.startswith("flow.decoder.estimator."))
    n_cfm  = len(decoder_keys)
    n_hift = len(mel2wav_keys)
    print(f"Wrote: encoder(+proj)~{n_flow} tensors, cfm={n_cfm}, hift={n_hift}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
