# qvac-chatterbox.cpp

**Chatterbox Turbo** (Resemble AI, MIT-licensed zero-shot text-to-speech)
ported to [`ggml`](https://github.com/ggml-org/ggml). Pure C++/ggml inference
on CPU / Metal / CUDA / Vulkan, with no runtime dependency on Python or
PyTorch.

Speed on a 10 s sentence (end-to-end, both T3 and S3Gen+HiFT):

| Backend                     | `gen_RTF` | Wall   | vs real-time |
|-----------------------------|----------:|-------:|-------------:|
| CPU (10-core EPYC, F16)     | 0.70      | 8.2 s  | 1.4×         |
| Metal (M3 Ultra, Q4_0)      | 0.14      | 2.0 s  | 7.3×         |
| Vulkan (RTX 5090, Q4_0)     | 0.06      | < 1 s  | 17.1×        |

See [`PROGRESS.md`](PROGRESS.md) for the full chronological development
journal, including numerical-parity stages and every optimization pass
that got us here (T3 Flash Attention, KV-cache layout rework, Metal
kernel patches, legacy Q4/Q5/Q8 quantization, etc.).

---

## Pipeline at a glance

```
      text                                                 24 kHz wav
       │                                                        ▲
       ▼                                                        │
  ┌────────────────────────────────────────────────────────────────┐
  │                       chatterbox                               │
  │                                                                │
  │   T3 (GPT-2 Medium)  ──►  S3Gen encoder  ──►  CFM (meanflow)   │
  │   text → speech toks      speech toks → h      h → mel         │
  │                                                                │
  │                          HiFT vocoder  ──►  24 kHz wav         │
  └────────────────────────────────────────────────────────────────┘
       ▲                                              ▲
   BPE tokenizer                               reference voice
   (embedded in T3 GGUF metadata)              (embedded in S3Gen GGUF)
```

One binary, one invocation, end to end — `scripts/synthesize.sh` is a
thin convenience wrapper that fills in the two GGUF paths.

## Prerequisites

- C++17 compiler (clang or gcc)
- cmake ≥ 3.14
- Python 3.10+ with `torch`, `numpy`, `gguf`, `safetensors`, `scipy`,
  `librosa`, `resampy` — needed **once**, at setup time only, to run the
  weight converters (which bake the precomputed mel filterbanks into the
  GGUFs) and the optional reference-dump scripts. Once the GGUFs exist,
  the C++ binary has zero runtime dependency on Python.

The easiest way to get the Python side is:

```bash
git clone https://github.com/resemble-ai/chatterbox.git chatterbox-ref
cd chatterbox-ref
python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install gguf safetensors scipy librosa resampy
cd -
```

## 1. Clone and build

```bash
# (from wherever you want the repo to live)
git clone git@github.com:gianni-cor/chatterbox.cpp.git
cd chatterbox.cpp

# ggml is vendored as a sibling subdirectory
git clone https://github.com/ggml-org/ggml.git ggml

# Apply our Metal op fixes (diag_mask_inf, pad_ext, faster conv_transpose_1d).
# Skip this if you're not building with -DGGML_METAL=ON.
(cd ggml && git apply ../patches/ggml-metal-chatterbox-ops.patch)

# Configure + build every target in one shot.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

To enable GPU acceleration, add the matching backend flag at configure
time: `-DGGML_METAL=ON` on Apple Silicon, `-DGGML_VULKAN=ON` on
Linux/Windows with a Vulkan loader, or `-DGGML_CUDA=ON` if you have the
CUDA toolkit. Pass `--n-gpu-layers 99` at runtime to actually use the
GPU. See `patches/README.md` for what the Metal patch does and why.

This produces the main binary plus a set of per-stage validation harnesses:

| Binary | What it does |
|--------|--------------|
| `build/chatterbox`            | End-to-end: text → speech tokens (T3) → wav (S3Gen + HiFT). Also handles voice cloning via `--reference-audio`. |
| `build/mel2wav`               | HiFT only: mel.npy → wav (demo) |
| `build/test-s3gen`            | Staged numerical validation of S3Gen encoder + CFM vs Python dumps |
| `build/test-resample`         | Round-trip SNR of the C++ Kaiser-windowed sinc resampler |
| `build/test-voice-features`   | 24 kHz 80-ch mel parity (prompt_feat) |
| `build/test-fbank`            | 16 kHz 80-ch Kaldi fbank parity |
| `build/test-voice-encoder`    | VoiceEncoder 256-d speaker embedding parity |
| `build/test-campplus`         | CAMPPlus 192-d embedding parity |
| `build/test-voice-embedding`  | wav → fbank → CAMPPlus end-to-end parity |
| `build/test-s3tokenizer`      | S3TokenizerV2 log-mel + speech-token parity |
| `build/test-metal-ops`        | Metal-only: parity check for `diag_mask_inf`, `pad_ext`, and fast `conv_transpose_1d` (only useful when built with `-DGGML_METAL=ON`) |

You'll normally only need `build/chatterbox`; the `test-*` binaries are
there for the staged-verification methodology in `PROGRESS.md`.

## 2. One-time: convert weights

```bash
# Activate the Python environment from the Prerequisites step
. ../chatterbox-ref/.venv/bin/activate

# Convert T3 weights + tokenizer + voice conditionals
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf

# Convert S3Gen encoder + CFM + HiFT weights
# (the built-in reference voice is embedded inside this GGUF)
python scripts/convert-s3gen-to-gguf.py --out models/chatterbox-s3gen.gguf
```

The scripts pull `ResembleAI/chatterbox-turbo` from Hugging Face Hub on
first run (about 1.5 GB). The BPE tokenizer (`vocab.json` + `merges.txt` +
`added_tokens.json`) is **embedded directly into the T3 GGUF** as
`tokenizer.ggml.*` metadata, so you don't need to keep those three files
around on disk.

You should now have:

```
models/
  chatterbox-t3-turbo.gguf   (~742 MB) — T3 GPT-2 Medium + embedded GPT-2 BPE
                               tokenizer + VoiceEncoder weights + built-in voice
  chatterbox-s3gen.gguf      (~1.0 GB) — S3Gen encoder/CFM + HiFT vocoder
                               + CAMPPlus speaker encoder + S3TokenizerV2
                               (everything needed for voice cloning, on top of
                               the built-in reference voice)
```

For numerical validation against PyTorch (optional, step 4), also run:

```bash
python scripts/dump-s3gen-reference.py \
  --text "Hello from ggml." --out artifacts/s3gen-ref \
  --seed 42 --n-predict 64 --device cpu
```

## 3. Run — end-to-end text → wav

The easiest way:

```bash
./scripts/synthesize.sh "Hello from native C plus plus." /tmp/out.wav
```

That's equivalent to running the binary directly:

```bash
./build/chatterbox \
  --model       models/chatterbox-t3-turbo.gguf \
  --s3gen-gguf  models/chatterbox-s3gen.gguf \
  --text        "Hello from native C plus plus." \
  --out         /tmp/out.wav
```

Everything is self-contained in the two `.gguf` files:

- `chatterbox-t3-turbo.gguf` embeds the BPE tokenizer (vocab + merges +
  added tokens) as standard `tokenizer.ggml.*` metadata, which the C++
  binary loads out of GGUF at startup.
- `chatterbox-s3gen.gguf` embeds the built-in reference voice (embedding,
  prompt token, prompt mel) under `s3gen/builtin/*`.

Advanced modes:

- **T3 only** — drop `--s3gen-gguf` + `--out`; write tokens with
  `--output tokens.txt`. Useful for piping into other tools.
- **S3Gen + HiFT only** — pass `--s3gen-gguf` + `--tokens-file FILE` with
  already-generated speech tokens and no `--model`.
- **Custom voice (voice cloning)** — point `--reference-audio` at a
  reference `.wav` and the C++ binary does everything else natively
  (no Python, no preprocessing step):

  ```bash
  ./build/chatterbox --model models/chatterbox-t3-turbo.gguf \
                     --s3gen-gguf models/chatterbox-s3gen.gguf \
                     --reference-audio me.wav \
                     --text "Hello in my voice." \
                     --out out.wav
  ```

  Requirements for the reference wav:
  - **Strictly more than 5 s** of clean mono speech (the binary enforces
    this and fails fast; 10–15 s gives the best similarity).
  - Any sample rate, any PCM bit-depth (binary resamples + downmixes).

  Loudness is normalised to **-27 LUFS** (ITU-R BS.1770-4 / EBU R 128)
  internally before preprocessing, so a quiet recording like a phone
  memo works as well as a studio track.  All five voice-conditioning
  tensors are produced in C++:

  | tensor                         | source                           |
  |--------------------------------|----------------------------------|
  | `speaker_emb`                  | C++ VoiceEncoder  (T3 GGUF)      |
  | `cond_prompt_speech_tokens`    | C++ S3TokenizerV2 (S3Gen GGUF)   |
  | `prompt_token`                 | C++ S3TokenizerV2 (S3Gen GGUF)   |
  | `embedding`                    | C++ CAMPPlus      (S3Gen GGUF)   |
  | `prompt_feat`                  | C++ mel extraction               |

- **Cache a voice for fast reuse (`--save-voice`)** — voice preprocessing
  (VoiceEncoder + CAMPPlus + S3TokenizerV2 + mel) adds ≈ 2 minutes on a
  Mac before every synthesis.  The five tensors don't depend on the
  text, so bake them once:

  ```bash
  # Bake the profile (no --text needed; just preprocesses + saves).
  ./build/chatterbox --model models/chatterbox-t3-turbo.gguf \
                     --s3gen-gguf models/chatterbox-s3gen.gguf \
                     --reference-audio me.wav \
                     --save-voice voices/me/
  # Writes voices/me/{speaker_emb, cond_prompt_speech_tokens,
  # embedding, prompt_token, prompt_feat}.npy (~160 KB total).

  # Reuse (≈ 17× faster; VoiceEncoder / CAMPPlus / S3TokenizerV2
  # / mel extraction are all skipped).
  ./build/chatterbox --model models/chatterbox-t3-turbo.gguf \
                     --s3gen-gguf models/chatterbox-s3gen.gguf \
                     --ref-dir voices/me/ \
                     --text "Anything you want." \
                     --out  out.wav
  ```

  You can mix the two: `--ref-dir D --reference-audio X.wav` will load
  any `.npy` present in `D` and compute the rest from `X.wav`.  Useful
  during development when you want to iterate on one tensor.

Play the result:

```bash
afplay /tmp/out.wav         # macOS
aplay  /tmp/out.wav         # Linux (alsa)
ffplay /tmp/out.wav         # any OS with ffmpeg
```

### Useful flags

- `--seed N` — change the RNG seed for the CFM initial noise and the SineGen
  excitation (same text, different voice "take").
- `--threads N` — override the default `std::thread::hardware_concurrency()`.
  The sweet spot on a 10-core CPU is 10.
- `--n-gpu-layers N` — move layers to the GPU backend when built with
  `-DGGML_METAL=ON` / `-DGGML_CUDA=ON` / `-DGGML_VULKAN=ON`.  Pass `99`
  (or any large number) to move everything.
- `--reference-audio PATH` — voice cloning input (see the Custom voice
  section above).
- `--save-voice DIR` — cache the five voice-conditioning tensors for
  reuse via `--ref-dir DIR`.
- `--ref-dir DIR` — load previously-baked voice tensors (or a subset)
  from `DIR/*.npy`.
- `--debug` (requires `--ref-dir`) — substitute Python-dumped reference
  values for the random bits so every stage can be bit-exactly compared
  to PyTorch.

Typical end-to-end timings on a 10 s sentence
(`gen_RTF = (T3_INFER_MS + S3GEN_INFER_MS) / AUDIO_MS`):

| Backend (weights)           | T3 gen  | S3Gen+HiFT gen | `gen_RTF` | Wall  | vs real-time |
|-----------------------------|--------:|---------------:|----------:|------:|-------------:|
| CPU (10-core EPYC, F16)     | 3998 ms | 2905 ms        | 0.70      | 8.2 s | 1.4×         |
| Metal (M3 Ultra, F16)       |  909 ms |  562 ms        | 0.15      | 2.1 s | 6.7×         |
| Metal (M3 Ultra, Q4_0)      |  886 ms |  608 ms        | 0.14      | 2.0 s | 7.3×         |
| Vulkan (RTX 5090, F16)      |  402 ms |  282 ms        | 0.06      | < 1 s | 15.6×        |
| Vulkan (RTX 5090, Q4_0)     |  347 ms |  284 ms        | 0.06      | < 1 s | 17.1×        |

`Wall` includes GGUF load time; `gen_RTF` is inference only.  Full
breakdown with CFM / HiFT sub-timings plus an ONNX baseline is in
[`PROGRESS.md §3.10 / §3.13`](PROGRESS.md).

Note: the binary also prints an inner `=== pipeline: … RTF=… ===` line
during synthesis.  That RTF covers **only the S3Gen + HiFT phase**
(it's the timer around `s3gen_synthesize_to_wav`, which runs after T3
is already done).  The table above reports the full end-to-end number.

## 4. Optional: validate against PyTorch

Every stage of the pipeline has a numerical regression test against
Python-dumped reference tensors:

```bash
./build/test-s3gen models/chatterbox-s3gen.gguf artifacts/s3gen-ref ALL
```

Expected output (rel error per stage):

```
Stage A  speaker_emb_affine    rel ≈ 1e-7
Stage B  input_embedded        rel = 0
Stage C  encoder_embed         rel ≈ 4e-7
Stage D  pre_lookahead         rel ≈ 3e-7
Stage E  enc_block0_out        rel ≈ 1e-7
Stage F  encoder_proj (mu)     rel ≈ 5e-7
Stage G1 time_mixer            rel ≈ 7e-7
Stage G2 cfm_resnet_out        rel ≈ 3e-7
Stage G3 tfm_out               rel ≈ 2e-7
Stage G4 cfm_step0_dxdt        rel ≈ 1e-6
Stage H1 f0                    rel ≈ 4e-6
Stage H3 conv_post             rel ≈ 6e-7
Stage H4 stft                  rel ≈ 8e-3 (boundary-bound)
Stage H5 waveform              rel ≈ 1e-4
```

For T3 bit-exact validation against the Python reference:

```bash
python scripts/reference-t3-turbo.py \
  --text "Hello from ggml." \
  --out-dir artifacts \
  --cpp-bin ./build/chatterbox \
  --cpp-model models/chatterbox-t3-turbo.gguf
```

## Repository layout

```
chatterbox.cpp/
  ggml/                          pristine ggml clone (not tracked)
  src/
    main.cpp                     CLI + T3 runtime            (chatterbox)
    chatterbox_tts.cpp           S3Gen + HiFT pipeline       (linked into chatterbox)
    s3gen_pipeline.h             public API for the S3Gen+HiFT back half
    mel2wav.cpp                  HiFT-only demo              (mel2wav)
    gpt2_bpe.{h,cpp}             self-contained GPT-2 BPE tokenizer

    voice_features.{h,cpp}       WAV I/O, sinc resampler, LUFS meter,
                                   24 kHz & 16 kHz log-mel extraction,
                                   Kaldi-style 80-ch fbank
    voice_encoder.{h,cpp}        3-layer LSTM → 256-d speaker_emb
                                   (matches Resemble VoiceEncoder)
    campplus.{h,cpp}              FunASR x-vector port (FCM + 3× CAMDense
                                   TDNN) → 192-d embedding
    s3tokenizer.{h,cpp}          6-layer FSMN-attn transformer + FSQ →
                                   25-Hz speech tokens
    dr_wav.h                     vendored single-header WAV reader
    npy.h                        minimal .npy load / save + compare

    test_*.cpp                   per-stage numerical-parity harnesses
  scripts/
    synthesize.sh                text → wav wrapper
    convert-t3-turbo-to-gguf.py  T3 weights + tokenizer + VE + builtin
                                   voice → T3 GGUF
    convert-s3gen-to-gguf.py     S3Gen encoder + CFM + HiFT + CAMPPlus +
                                   S3TokenizerV2 + mel filterbanks →
                                   S3Gen GGUF
    dump-*-reference.py          PyTorch → .npy intermediates for the
                                   per-stage harnesses
    reference-t3-turbo.py        PyTorch T3 bit-exact compare vs C++
    compare-tokenizer.py         10-case BPE tokenizer compare vs HF
  patches/
    ggml-metal-chatterbox-ops.patch  ggml-metal fixes (see patches/README.md)
    README.md                    applies-to / what-it-does notes
  voices/                        baked voice profiles (not tracked; populated
                                   by --save-voice)
  models/                        generated GGUFs (not tracked)
  artifacts/                     .npy dumps for validation (not tracked)
  CMakeLists.txt                 top-level build
  README.md                      this file
  PROGRESS.md                    chronological development journal
```

## Troubleshooting

**`error: this GGUF has no embedded tokenizer`** — you're running against
a legacy T3 GGUF built before the tokenizer was embedded. Re-run the
converter to produce a fresh GGUF:

```bash
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
```

**`--debug requires --ref-dir`** — debug mode substitutes Python-dumped
random bits to make every intermediate tensor bit-exactly comparable.
Run `python scripts/dump-s3gen-reference.py --out artifacts/s3gen-ref …`
first, then pass `--ref-dir artifacts/s3gen-ref`.

**Output is much louder than the Python reference** — expected: the Python
reference dump uses a very short utterance (mostly silence). Generate a
longer sentence and compare RMS. Differences up to ~2.5 % in spectrogram
magnitude are from the stochastic SineGen excitation (non-bit-exact RNG
between `std::mt19937` and `torch.rand`).

**Slower than real-time** — make sure you built `-DCMAKE_BUILD_TYPE=Release`
and that `--threads` picks up all your cores. The binary defaults to
`std::thread::hardware_concurrency()`.
