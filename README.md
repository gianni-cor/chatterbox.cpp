# qvac-chatterbox.cpp

**Chatterbox Turbo** (Resemble AI, MIT-licensed zero-shot text-to-speech)
ported to [`ggml`](https://github.com/ggml-org/ggml). Pure C++/ggml inference
on CPU (Linux/macOS) with no runtime dependency on Python or PyTorch.

See [`PROGRESS.md`](PROGRESS.md) for the full chronological development
journal, including verification stages, numerical parity results, and the
optimization pass that got us to **3.6× faster than real-time on CPU**.

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
- Python 3.10+ with `torch`, `numpy`, `gguf`, `safetensors`, `scipy` —
  needed **once** for the weight conversion and the reference-voice dump.

The easiest way to get the Python side is:

```bash
git clone https://github.com/resemble-ai/chatterbox.git chatterbox-ref
cd chatterbox-ref
python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install gguf safetensors scipy
cd -
```

## 1. Clone and build

```bash
# (from wherever you want the repo to live)
git clone git@github.com:gianni-cor/chatterbox.cpp.git
cd chatterbox.cpp

# ggml is vendored as a sibling subdirectory
git clone https://github.com/ggml-org/ggml.git ggml

# Build all 3 binaries: chatterbox, mel2wav, test-s3gen
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

This produces:

| Binary | What it does |
|--------|--------------|
| `build/chatterbox` | End-to-end: text → speech tokens (T3) → wav (S3Gen + HiFT) |
| `build/mel2wav` | HiFT only: mel.npy → wav (demo) |
| `build/test-s3gen` | Staged numerical validation vs Python dumps |

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
  chatterbox-t3-turbo.gguf   (~730 MB, F16 T3 weights + embedded GPT-2 BPE tokenizer)
  chatterbox-s3gen.gguf      (~410 MB, F32 S3Gen + HiFT weights + built-in voice)
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
- **Custom voice (voice cloning)** — first turn a reference `.wav` into
  a voice profile directory, then pass it as `--ref-dir`:

  ```bash
  python scripts/prepare-voice.py --ref-audio me.wav --out voices/me/
  ./build/chatterbox --model models/chatterbox-t3-turbo.gguf \
                     --s3gen-gguf models/chatterbox-s3gen.gguf \
                     --ref-dir voices/me/ \
                     --reference-audio me.wav \
                     --text "Hello in my voice." \
                     --out out.wav
  ```

  `voices/me/` holds five `.npy` tensors (speaker embedding + prompt
  tokens + prompt mel for T3 and S3Gen respectively). Reference audio
  needs to be at least 5 s of clean speech; longer helps.

  `--reference-audio me.wav` is optional: when set, the C++ binary
  computes `prompt_feat` natively (WAV load + Kaiser-windowed sinc
  resample to 24 kHz + 80-channel log-mel) and ignores
  `voices/me/prompt_feat.npy`. The other four voice tensors still come
  from the Python helper for now; native `VoiceEncoder` / `CAMPPlus` /
  `S3TokenizerV2` are on the backlog (`PROGRESS.md` A1 phases 2c-2e).

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
- `--debug` (requires `--ref-dir`) — substitute Python-dumped reference
  values for the random bits so every stage can be bit-exactly compared to
  PyTorch.

Typical timings on a 10-core EPYC CPU:

```
>>> [1/2] T3: text -> speech tokens
    generated 145 speech tokens
>>> [2/2] S3Gen + HiFT: speech tokens -> wav
Using 10 threads
  [encoder] 286 ms
  [cfm_total] 785 ms
  [hift_total] 1312 ms
=== pipeline: 2383 ms for 8640 ms of audio (RTF=0.28, 3.6x faster than real-time) ===
```

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
    main.cpp                     CLI + T3 runtime      (chatterbox)
    chatterbox_tts.cpp           S3Gen + HiFT pipeline (linked into chatterbox)
    s3gen_pipeline.h             public API for the S3Gen+HiFT back half
    mel2wav.cpp                  HiFT-only demo        (mel2wav)
    test_s3gen.cpp               staged validation     (test-s3gen)
    gpt2_bpe.{h,cpp}             self-contained GPT-2 BPE tokenizer
    npy.h                        minimal .npy loader + compare helpers
  scripts/
    synthesize.sh                text → wav wrapper
    convert-t3-turbo-to-gguf.py  T3 weights + conds → GGUF
    convert-s3gen-to-gguf.py     flow (encoder + CFM) + HiFT → GGUF
    dump-s3gen-reference.py      PyTorch → .npy intermediates
    reference-t3-turbo.py        PyTorch T3 bit-exact compare vs C++
    compare-tokenizer.py         10-case BPE tokenizer compare vs HF
  models/                        generated GGUFs (not tracked)
  artifacts/s3gen-ref/           generated .npy reference tensors (not tracked)
  CMakeLists.txt                 top-level build: add_subdirectory(ggml) + 3 targets
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
