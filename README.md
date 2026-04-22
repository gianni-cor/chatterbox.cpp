# chatterbox.cpp

**Chatterbox Turbo** (Resemble AI, MIT-licensed zero-shot text-to-speech)
ported to [`ggml`](https://github.com/ggml-org/ggml). Pure C++/ggml inference
on CPU / Metal / CUDA / Vulkan, with no runtime dependency on Python or
PyTorch.

End-to-end inference on a short sentence with voice cloning from an 11 s
reference wav (T3 + S3Gen + HiFT, warm runs, excludes model load):

| Backend                              | Wall      | `RTF`  | vs real-time | vs ONNX Runtime |
|--------------------------------------|----------:|-------:|-------------:|----------------:|
| Vulkan (RTX 5090, Q4_0)              |   463 ms  | 0.07   | **14.2×**    | **13.8× faster** |
| Metal (Mac Studio M3 Ultra, Q4_0)    |   985 ms  | 0.16   | 6.4×         | 17.5× faster    |
| CPU (AMD Ryzen 9 9950X, AVX, Q4_0)   | 5 397 ms  | 0.82   | 1.2×         | 1.2× faster     |
| CPU (Mac Studio M3 Ultra, NEON)      | 7 568 ms  | 1.05   | 0.96×        | 2.3× faster     |
| Reference (ONNX Runtime, CPU Q4)     | 6.4–17 s  | 1.2–3.2 | 0.3–0.85×   | —               |

See the [full benchmark](#performance) section below for the per-stage
breakdown, or [`PROGRESS.md`](PROGRESS.md) for the full chronological
development journal — every numerical-parity stage and optimization pass
(T3 Flash Attention, KV-cache layout rework, Metal kernel patches,
CAMPPlus + VoiceEncoder + S3TokenizerV2 ported to ggml graphs, mel
extraction via STFT matmul, legacy Q4/Q5/Q8 quantization, etc.).

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

# Clone ggml at the pinned commit and apply our Metal op patches.
# Skip the patch part (i.e. just `git clone ... ggml`) if you're not
# building with -DGGML_METAL=ON.
./scripts/setup-ggml.sh

# Configure + build every target in one shot.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

`scripts/setup-ggml.sh` is idempotent: it clones upstream
[`ggml`](https://github.com/ggml-org/ggml) into `./ggml`, checks out the
commit our patch is pinned against (`GGML_COMMIT` at the top of the
script), and applies
[`patches/ggml-metal-chatterbox-ops.patch`](patches/ggml-metal-chatterbox-ops.patch).
Re-running it is a no-op; bump the pinned commit inside the script
whenever the patch is re-generated against a newer upstream.

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

### Optional: quantize the models (smaller + faster)

Both GGUFs can be quantized to `Q8_0` (near-lossless) or `Q4_0` (light
quality loss, 3× smaller).  T3 supports `--quant` in its converter; the
S3Gen GGUF is re-quantized with a small standalone tool because
`llama-quantize` doesn't know the `chatterbox-s3gen` architecture.

```bash
# T3: quantize at conversion time.  Choose q8_0 / q5_0 / q4_0.
python scripts/convert-t3-turbo-to-gguf.py \
  --out models/t3-q8_0.gguf --quant q8_0
python scripts/convert-t3-turbo-to-gguf.py \
  --out models/t3-q4_0.gguf --quant q4_0

# S3Gen: re-quantize the already-converted F32 GGUF in place.
python scripts/requantize-s3gen.py \
  models/chatterbox-s3gen.gguf \
  models/chatterbox-s3gen-q8_0.gguf q8_0

python scripts/requantize-s3gen.py \
  models/chatterbox-s3gen.gguf \
  models/chatterbox-s3gen-q4_0.gguf q4_0
```

Measured on the QVAC paragraph (M3 Ultra, Metal, streaming mode
`--stream-chunk-tokens 25 --max-sentence-chars 100`):

| T3 / S3Gen                | total size | first-audio | total wall | cos sim¹ |
|---------------------------|-----------:|------------:|-----------:|--------:|
| F16 / F32 (baseline)      |  1 757 MB  |  1 604 ms   |   28.6 s   |  1.000  |
| Q8_0 / F32                |  1 476 MB  |  1 451 ms   |   27.2 s   |   —     |
| F16 / Q8_0                |  1 532 MB  |  1 646 ms   |   28.2 s   |  0.991  |
| **Q8_0 / Q8_0**           | **1 251 MB** | **1 399 ms** | **26.4 s** | 0.991 |
| **Q4_0 / Q4_0**           | **1 071 MB** | **1 510 ms** | **26.7 s** | 0.66²  |

¹ Cosine similarity of the final waveform vs the F16/F32 baseline.

² Q4_0 quantization shifts the CFM diffusion ODE's trajectory enough
  to land on a *different sample* from the same noise seed.  Subjective
  quality is essentially the same (in-distribution speech, correct
  phonemes, stable voice); it's just a different legitimate sample
  rather than a lower-fidelity version of the baseline.

Using both Q8_0 variants cuts **~500 MB off disk**, drops first-audio
latency **~13 %**, speeds total wall-clock **~8 %**, and produces
audibly-identical output (cos-sim > 0.99 vs F32 reference waveform).
Q4_0 trims another ~180 MB on top for roughly the same speed — best
choice on memory-constrained targets (mobile, low-end CPUs) when you
don't need per-seed reproducibility against the F32 baseline.

Note: the S3Gen requantize script only compresses the 385 big 2-D
matmul weights (encoder attention/MLPs + CFM projections + flow FFs).
The 1 664 other tensors — biases, norms, spectral filterbanks, the
input-embedding table, the 3-D convolution weights — remain at their
source dtype to keep numerics clean.  That's why Q4_0 ends up only
~15 % smaller than Q8_0 rather than 2× smaller; the bulk not covered
by block quantization dominates.

Pass the quantized GGUFs to `chatterbox` exactly like the defaults:

```bash
./build/chatterbox \
  --model      models/t3-q8_0.gguf \
  --s3gen-gguf models/chatterbox-s3gen-q8_0.gguf \
  --text "Hello from the quantized port." \
  --n-gpu-layers 99 --out out.wav
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

<a id="performance"></a>
## Performance

Reproducible perf check vs an ONNX Runtime Q4 baseline (same
architecture) on the same machine.  Shared setup:

- Text: *"Hello from native C plus plus. This audio was generated end
  to end on CPU using ggml."*
- Reference voice: `test/reference-audio/jfk.wav` (11 s mono 16 kHz)
- Seed: 42, warm 3-run average, inference only (excludes model load)

### Mac Studio M3 Ultra (96 GB unified memory)

| Implementation                        | Backend         | T3 gen             | S3Gen+HiFT gen | Total inference | RTF   | vs real-time |
|---------------------------------------|-----------------|-------------------:|---------------:|----------------:|------:|-------------:|
| **`chatterbox.cpp` Q4_0**             | **Metal**       |  573 ms / 155 tok  |    412 ms      |   **985 ms**    | 0.16  | **6.4×**     |
| `chatterbox.cpp` Q4_0                 | CPU (NEON+Accel)| 2 045 ms / 178 tok |  5 523 ms      |    7 568 ms     | 1.05  | 0.96×        |
| ONNX Runtime Q4 baseline              | CPU             |        —           |      —         |   17 190 ms     | 3.18  | 0.31×        |

`chatterbox.cpp` (Metal) is **17.5× faster than ONNX Runtime** on the
same machine; the CPU-only build is still 2.3× faster.

### Linux RTX 5090 + AMD Ryzen 9 9950X

| Implementation                        | Backend         | T3 gen             | S3Gen+HiFT gen | Total inference | RTF   | vs real-time |
|---------------------------------------|-----------------|-------------------:|---------------:|----------------:|------:|-------------:|
| **`chatterbox.cpp` Q4_0**             | **Vulkan**      |  241 ms / 161 tok  |    222 ms      |    **463 ms**   | 0.07  | **14.2×**    |
| `chatterbox.cpp` Q4_0                 | CPU (AVX)       | 2 161 ms / 161 tok |  3 236 ms      |    5 397 ms     | 0.82  | 1.2×         |
| ONNX Runtime Q4 baseline              | CPU             |        —           |      —         |    6 373 ms     | 1.18  | 0.85×        |

`chatterbox.cpp` (Vulkan) is **13.8× faster than ONNX Runtime** on the
same machine.  Note that the ONNX Runtime baseline here only uses the
CPU execution provider; a CUDA build would narrow the gap, but is not
included in this comparison.

### Per-stage S3Gen + HiFT breakdown (GPU builds)

| Stage           | M3 Ultra Metal  | RTX 5090 Vulkan |
|-----------------|----------------:|----------------:|
| T3 per token    | 3.70 ms / tok   | **1.50 ms/tok** |
| encoder         |    38 ms        |    35 ms        |
| cfm_step0       |    69 ms        |    84 ms        |
| cfm_step1       |    49 ms        |    13 ms        |
| cfm_total       |   124 ms        |   100 ms        |
| f0_predictor    |   3.1 ms        |   1.1 ms        |
| sinegen (CPU)   |    15 ms        |    16 ms        |
| stft            |   3.1 ms        |   1.0 ms        |
| **hift_decode** |   225 ms        |    66 ms        |
| hift_total      |   246 ms        |    84 ms        |

HiFT `conv_transpose_1d` upsampling is the single biggest stage on
Metal today; the 5090 chews through it 3.4× faster, which is where the
remaining end-to-end gap comes from.

### Reproducing these numbers

```bash
# Build chatterbox.cpp, then:
./build/chatterbox \
    --model       models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf  models/chatterbox-s3gen.gguf \
    --reference-audio test/reference-audio/jfk.wav \
    --text "Hello from native C plus plus. This audio was generated end to end on CPU using ggml." \
    --out /tmp/bench.wav \
    --seed 42 \
    --n-gpu-layers 99   # 0 or omit for CPU
```

The binary prints both the per-stage timings and `BENCH:` lines that
scripts can scrape.  Note: the binary also prints an inner
`=== pipeline: … RTF=… ===` line — that RTF covers **only the
S3Gen + HiFT phase** (the timer around `s3gen_synthesize_to_wav`, which
runs after T3 is already done).  The tables above report the full
end-to-end number (T3_INFER + S3GEN_INFER).

`gen_RTF = (T3_INFER_MS + S3GEN_INFER_MS) / AUDIO_MS`

Token counts vary slightly across backends because the CPU-side
sampler reads logits that come out of different float-reduction orders
per backend; per-token T3 cost is the directly-comparable figure.
Full development history and older backend combinations (F16 vs
Q4_0 / Q5_0 / Q8_0, plus other machines) are in
[`PROGRESS.md §3.10 / §3.13`](PROGRESS.md).

### Streaming mode — low-latency playback

For interactive use cases, the binary can emit audio **chunk-by-chunk**
as it's generated instead of waiting for the whole sentence to finish.
Any non-zero `--stream-chunk-tokens N` turns streaming on.

**Flags:**

- `--stream-chunk-tokens N` — main knob; N speech tokens per chunk
  (25 ≈ 1 s of audio, 50 ≈ 2 s).
- `--stream-first-chunk-tokens N` — override the *first* chunk's size
  so first-audio-out lands early while later chunks stay big and keep
  overall RTF low.  Typical: 10.
- `--stream-cfm-steps N` — CFM Euler step count.  Default 2 (matches
  Python meanflow).  `1` halves CFM cost with a small quality penalty;
  Turbo's meanflow training makes 1-step a valid sampling mode per the
  paper.
- `--out -` — emit raw `s16le` mono @ 24 kHz to stdout instead of
  writing a wav file, so the output can be piped straight into a
  player.

**Recommended low-latency preset for interactive use:**

```bash
brew install sox      # one-time, for the `play` command

./build/chatterbox \
    --model      models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf models/chatterbox-s3gen.gguf \
    --text       "Hello from streaming Chatterbox." \
    --stream-first-chunk-tokens 10 \
    --stream-chunk-tokens       25 \
    --stream-cfm-steps          1 \
    --n-gpu-layers              99 \
    --out - \
  | play -q -t raw -r 24000 -b 16 -e signed -c 1 -
```

`play` ships with `sox` and routes straight to CoreAudio.  If you
prefer, the same stdout stream works with `ffplay -f s16le -ar 24000
-ch_layout mono -nodisp -i -` or piped through a Python
`sounddevice.play()` one-liner; on some macOS 26 builds ffplay's SDL
output is silent for raw piped audio, so `sox play` is the safest
default.

You can also drop the `--out -` to get a regular wav:

```bash
./build/chatterbox … --stream-chunk-tokens 50 --out out.wav
afplay out.wav
```

**Latency and throughput** on an Apple M4 with the Metal backend and
the preset above, feeding the sentence *"Hello from streaming
Chatterbox, I am John and I work in Google since 2010. I love to go
out with my friends, eat some pizza and also drink some wine. I also
love to travel around the world alone."* (produces 317 speech tokens,
~12.7 s of audio):

| metric | value |
|---|---|
| first-audio-out latency | **279 ms** |
| chunk 1 (10-token bootstrap) | RTF 0.99 |
| chunks 2–13 (steady-state, 25 tokens each) | **RTF 0.30 – 0.63** |
| chunk 14 (tail finalise) | RTF 1.42 |
| total wall time | 11.5 s for 12.7 s of audio |
| overall RTF | **0.90** |

The steady-state RTFs stay comfortably below 1.0, so the streamer
sustainably pushes audio faster than real-time playback consumes it.
Chunk 1 is small by design so first audio lands in ~280 ms; the final
chunk is short and relatively slow (fixed encoder/CFM overhead
amortised over only 0.4 s of audio).

For the full journal of how streaming got there — bit-exact CFM parity,
`cache_source` + `trim_fade` port, `--out -` stdout wiring, per-chunk
tuning — see [`PROGRESS.md §B1`](PROGRESS.md).

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

## License

Released under the [MIT License](LICENSE) — Copyright (c) 2026 Gianfranco
Cordella. The bundled `ggml/` is also MIT-licensed
([ggml/LICENSE](ggml/LICENSE)). The upstream Python implementation
([Chatterbox](https://github.com/resemble-ai/chatterbox), Copyright (c) 2025
Resemble AI) is likewise MIT-licensed; see `LICENSE` for the third-party
attribution block.
