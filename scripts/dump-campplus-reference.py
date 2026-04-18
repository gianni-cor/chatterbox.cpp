#!/usr/bin/env python3
"""Dump CAMPPlus inputs and outputs for numerical validation against the
C++ port (src/campplus.cpp).

Writes:
  fbank.npy      — (T, 80) mean-subtracted Kaldi-fbank at 16 kHz.  This is
                   exactly what the CAMPPlus.forward pass receives.
  embedding.npy  — (192,) raw CAMPPlus output (pre-L2-norm).  Matches the
                   `embedding` tensor that prepare-voice.py stores.

Example:

    . ~/chatterbox-ref/.venv/bin/activate
    python scripts/dump-campplus-reference.py REF.wav --out /tmp/camp_ref
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path, help="Reference wav (any SR; resampled to 16 kHz)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir; creates fbank.npy + embedding.npy inside it")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from chatterbox.models.s3gen.xvector import extract_feature

    tts = ChatterboxTurboTTS.from_pretrained("cpu")
    speaker_encoder = tts.s3gen.speaker_encoder
    speaker_encoder.eval()

    wav, sr = torchaudio.load(str(args.wav))
    wav = wav.mean(dim=0) if wav.ndim == 2 and wav.shape[0] > 1 else wav.squeeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # This is exactly what speaker_encoder.inference does.
    feats, lens, times = extract_feature([wav])
    # feats: (1, T, 80) after per-utterance mean subtraction.
    fbank = feats[0].numpy().astype(np.float32)

    with torch.no_grad():
        emb = speaker_encoder.forward(feats.to(torch.float32))
    emb = emb[0].cpu().numpy().astype(np.float32)

    np.save(args.out / "fbank.npy", np.ascontiguousarray(fbank))
    np.save(args.out / "embedding.npy", np.ascontiguousarray(emb))
    print(f"fbank.npy      shape={fbank.shape}  dtype={fbank.dtype}")
    print(f"embedding.npy  shape={emb.shape}  dtype={emb.dtype}  norm={np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    main()
