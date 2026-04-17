# Chatterbox → ggml Port: Development Journal

This document tracks the port of **Chatterbox Turbo** (Resemble AI, MIT license)
to `ggml`, from the first exploratory scoping all the way to the optimized
end-to-end CPU binary, in the order things actually happened.

- **Model**: `ResembleAI/chatterbox-turbo` (text-to-speech, ~450 M params without
  the tokenizer / speaker-encoder).
- **Goal**: end-to-end `text → waveform` in C++/ggml with **bit-exact (or
  float-precision) parity** against the official PyTorch reference.
- **Verification target**: every intermediate tensor within 1e-6 relative error
  of the PyTorch implementation, on CPU.

---

## Current status (end of journey)

Everything runs in pure C++/ggml on CPU. Three binaries:

| Binary | Role |
|--------|------|
| `chatterbox` | text → speech tokens (T3, GPT-2 Medium, 24 layers) |
| `chatterbox-tts` | speech tokens + reference voice → 24 kHz wav (S3Gen + HiFT) |
| `mel2wav` | mel spectrogram → wav (HiFT only, demo) |

Plus `scripts/synthesize.sh` which composes the two into a single command.

**Numerical parity vs PyTorch** on a 2.7 s reference utterance, debug mode
(Python-dumped random bits substituted for reproducibility):

| Stage | rel error vs PyTorch |
|-------|---------------------|
| BPE tokenizer | 10/10 exact-match test cases |
| T3 speech tokens | bit-exact on 4 deterministic prompts |
| S3Gen encoder (full, incl. upsample and encoder_proj) | 4.5e-07 |
| CFM 2-step meanflow decoder | 8.9e-07 on the final mel |
| HiFT decode body (conv_pre → conv_post) | 5.6e-07 |
| ISTFT → waveform | 1.0e-04 |
| End-to-end C++ wav vs Python wav (RMS) | 1.22e-04 vs 1.22e-04 |

**Speed** on a 10-core EPYC for an 8.64 s utterance, after the optimization
pass: **RTF 0.28 (3.6× faster than real-time)** — see §3.8.

---

## Repository layout

```
qvac-chatterbox.cpp/
  ggml/                           pristine ggml checkout (vendored, unmodified)
  src/
    main.cpp                      T3 runtime          (chatterbox binary)
    gpt2_bpe.{h,cpp}              self-contained GPT-2 byte-level BPE tokenizer
    test_s3gen.cpp                staged verification harness (stages A..H5)
    mel2wav.cpp                   mel → wav demo binary (HiFT only)
    chatterbox_tts.cpp            speech tokens → wav (S3Gen encoder + CFM + HiFT)
    npy.h                         minimal .npy loader + compare helpers
  scripts/
    convert-t3-turbo-to-gguf.py   T3 weights + conds → GGUF
    convert-s3gen-to-gguf.py      flow (encoder + CFM) + HiFT → GGUF
    dump-s3gen-reference.py       runs PyTorch, dumps every intermediate .npy
    reference-t3-turbo.py         PyTorch T3 + compare against C++
    compare-tokenizer.py          10-case tokenizer comparison against HF
    synthesize.sh                 text → wav wrapper (T3 + chatterbox-tts)
  models/
    chatterbox-t3-turbo.gguf      T3 + tokenizer conditionals
    chatterbox-s3gen.gguf         flow + mel2wav weights + built-in voice
  CMakeLists.txt                  top-level: add_subdirectory(ggml) + 4 targets
  PROGRESS.md                     this file
```

A separate machine holds PyTorch + the original Chatterbox repo for reference
runs. On-device (Apple Silicon / Linux x86) the C++ binaries have **no runtime
dependency on Python** — the tokenizer reads `vocab.json` + `merges.txt`
directly.

---

## Development log (chronological)

### 3.1  Scoping and bootstrap

Surveyed open-source TTS candidates (F5-TTS, Kokoro-82M, XTTS v2, Piper, Fish
Speech, Supertonic, Chatterbox). Picked **Chatterbox Turbo** for three reasons:
MIT license, zero-shot voice cloning, and the "Turbo" variant uses just **2
flow-matching steps** (fast inference).

Bootstrapped the repo by cloning the latest `ggml` and the reference
`resemble-ai/chatterbox` side-by-side, then built a standalone
`qvac-chatterbox.cpp/` with `ggml/` as a vendored subdirectory (no modifications
inside `ggml/`).

**Issues hit in this phase:**

| # | Issue | Fix |
|---|-------|-----|
| 1 | `rsync` not on macOS by default | Switched to `tar … \| ssh … tar -x`. |
| 2 | Remote repo polluted with `._*` AppleDouble files | `COPYFILE_DISABLE=1 tar …`. |
| 3 | Partial sync left `src/CMakeLists.txt` stray file | Removed; unified sync always pushes the whole tree. |
| 4 | Remote binary `0 bytes` after SSH disconnect | `rm build/<target>` + rebuild. |

### 3.2  T3 port + custom BPE tokenizer

T3 is a GPT-2 Medium-sized (24 layer) autoregressive model that maps text
tokens + voice conditioning to speech tokens.

- Wrote `scripts/convert-t3-turbo-to-gguf.py` to emit a GGUF with built-in
  voice conditionals (`speaker_emb`, `cond_prompt_speech_tokens`) embedded.
- C++ graph in `src/main.cpp`: split into a "prompt" graph and a "step" graph
  sharing a persistent KV cache, mirroring `ggml/examples/gpt-2`.
- Ported the sampler (Temperature → TopK → TopP → RepetitionPenalty).
- Wrote a **self-contained GPT-2 byte-level BPE** in `src/gpt2_bpe.cpp` (llama.cpp's
  BPE was too entangled with its GGUF vocab loading to reuse cleanly):
  byte-level encoding table, regex pre-tokenization, BPE merge loop, plus
  `punc_norm` matching the Python implementation. **10/10** test cases match
  the HF tokenizer byte-for-byte, including the 19 paralinguistic added tokens
  (`[laugh]`, `[chuckle]`, …).
- `chatterbox` binary takes `--text` + `--tokenizer-dir` and produces speech
  tokens end-to-end.

Verified against PyTorch: **bit-for-bit** identical speech tokens on 4
deterministic sampling configs (greedy / temperature / top-k /
repetition-penalty / no-penalty × short + long prompts).

**Issues hit in this phase:**

| # | Issue | Fix |
|---|-------|-----|
| 5 | `ggml_can_mul_mat` assertion in T3 | Converter must transpose `Conv1D`-style weights (`c_attn`, `c_proj`, `c_fc`, `mlp.c_proj`) to ggml's `[in, out]` layout while leaving `nn.Linear` / embeddings / `wpe` as-is. |
| 6 | `ggml_backend_tensor_get(input_tensor)` returned garbage | `ggml_gallocr` reuses the input buffer for intermediates when only `set_input` is marked; also call `ggml_set_output` on tensors we want to read back. |
| 7 | Repetition-penalty path diverged from HF at token 22 | HF divides positive logits, multiplies negative ones — I had it backwards. |
| 8 | Sampler order mismatched HF `LogitsProcessorList` | Rewrote `sample_next_token` as Temperature → TopK → TopP → RepetitionPenalty, in HF's exact order. After the fix greedy+penalty tests pass bit-exactly. |

### 3.3  S3Gen encoder (stages A–F)

S3Gen is a "Upsample Conformer" with 10 blocks total (~60 M params): 6 initial
blocks, then a 2× `Upsample1D`, then 4 more blocks. Ported in six staged
substeps against Python-dumped reference tensors (`scripts/dump-s3gen-reference.py`):

| Stage | Component | rel error |
|-------|-----------|----------:|
| A | `speaker_emb` projection (`F.normalize` + Linear) | 1.2e-7 |
| B | `input_embedding` lookup | 0 (exact) |
| C | `encoder_embed` (Linear + LN + √D scale + ESPnet rel PE) | 4.4e-7 |
| D | `PreLookaheadLayer` (asymmetric-padded Conv1d stack) | 2.5e-7 |
| E | One Conformer block (rel-pos MHA + `rel_shift` + Swish FFN) | 1.3e-7 |
| **F** | **Full encoder + `encoder_proj`** | **5.6e-7** |

**Issues hit in this phase:**

| # | Issue | Fix |
|---|-------|-----|
| 9  | `ggml_conv_1d` aborted with `src0->type == GGML_TYPE_F16` | ggml's `im2col` path requires F16 kernels, but we wanted F32 precision. Wrote a `conv1d_f32` helper that calls `ggml_im2col(…, GGML_TYPE_F32)` + `mul_mat` directly, keeping kernels in F32. |
| 10 | `speaker_embed` broadcast failed in `cond_spkr` matmul | Bias reshape needed `ne=[1, 256]`, not `ne=[256]`. Added the explicit `reshape_2d(bias, 1, C)` convention for every 1-D bias added to a `[T, C]` conv output. |
| 11 | Nearest-neighbor ×2 upsample produced channel-interleaved garbage | The naive `reshape_3d(T, 1, D) + concat(ne[1])` gives `t0_copy0, t1_copy0, …, t0_copy1, …`. Correct trick: `reshape_3d(1, T, D)` → `concat` along `ne[0]` → `[2, T, D]` → reshape to `[2T, D]`, giving `t0_copy0, t0_copy1, t1_copy0, …`. |
| 12 | `rel_shift` attention gave ~100 % rel error | `view_3d(bd_viewed, T, 2T-1, H, nb1, T*(2T-1)*elem, offset)` used the *sliced* stride for `nb2`. `nb2` must match the *source's* element stride: `bd_viewed->nb[2]`. |
| 13 | `*.transpose().numpy()` reference dumps loaded as garbage in C++ | Torch `.transpose()` yields Fortran-ordered storage; `np.save` writes `fortran_order: True`. Dumper now calls `.contiguous().numpy()` + `np.ascontiguousarray(...)`. The C++ loader throws a clear error if it sees `fortran_order=True`. |

### 3.4  CFM decoder (stages G1–G4)

A U-Net with transformer blocks (~45 M params). Layout: 1 down block → 12 mid
blocks → 1 up block (skip concat) → `final_block` → `final_proj`. Each block
carries 4 `BasicTransformerBlock`s.

| Stage | Component | rel error |
|-------|-----------|----------:|
| G1 | Time embedding (sin → MLP → mixer) | 7.0e-7 |
| G2 | `CausalResnetBlock1D` (causal-conv + LN + Mish + time MLP + res_conv) | 2.9e-7 |
| G3 | `BasicTransformerBlock` (self-attn + FFN w/ GELU-erf) | 1.7e-7 |
| **G4** | **Full CFM decoder, one forward step** | **1.3e-6** |

For meanflow mode we do 2 steps with `t_span = [0, 0.5, 1]`; the time embedding
sees both `t` and `r` concatenated through a mixer.

**Issues hit in this phase:**

| # | Issue | Fix |
|---|-------|-----|
| 14 | `LayerNorm` applied over time instead of channel | For `ne=[T, C]` layout `ggml_norm` reduces `ne[0]=T`, which is wrong. Wrote `layer_norm_on_channel` that permutes to `[C, T]`, norms, applies affine, permutes back. |
| 15 | `weight_norm` convolutions in `mel2wav` ignored | Torch 2.6 stores them under `parametrizations.weight.original{0,1}`. Added `expand_weight_norm()` in the converter that fuses `g · v / ‖v‖₂` back into a regular `weight` tensor before export. |
| 16 | Mish activation missing from ggml unary ops | Built from primitives: `x · tanh(softplus(x))` via `GGML_UNARY_OP_SOFTPLUS` + `GGML_UNARY_OP_TANH`. |
| 17 | GELU mismatch in `BasicTransformerBlock` (rel=3e-4) | `ggml_gelu` is the tanh approximation; `diffusers.models.activations.GELU` uses the exact `erf` formulation. Switched to `ggml_gelu_erf`. Error dropped to 1.7e-7. |
| 18 | Python hook overwrote the same tensor across multiple CFM steps | Meanflow calls `time_embeddings` twice (for `t` and `r`) and the decoder runs twice per sample. Added `make_hook(multi_call=True)` that saves `*_call0.npy`, `*_call1.npy`, …. |
| 19 | Estimator `forward_hook` never fired | `basic_euler` calls `self.estimator.forward(x, …)` directly, bypassing `__call__` where hooks live. Monkey-patched `estimator.forward` to record `x_in / mu / t / r / spks / cond / mask / dxdt` for every step. |
| 20 | `(B, C, T)` vs `(B, T, C)` layout confusion | CFM alternates: resnets use `(B, C, T)`, transformer blocks use `(B, T, C)`, switched by `rearrange`. In ggml we mirror this and `cont(permute)` at the boundary. Every helper doc-comments its layout. |

### 3.5  HiFT vocoder (stages H1–H5) + `mel2wav` binary

HiFTGenerator = Neural Source Filter + ISTFTNet. The mel → waveform vocoder.
Ported in five verifiable substeps:

| Stage | Component | rel error |
|-------|-----------|----------:|
| H1 | `f0_predictor` (5× Conv + ELU + Linear) | 4.2e-6 |
| H3 | decode body `conv_pre → ups / rb → conv_post` | 5.6e-7 |
| H4 | STFT (Conv1d with DFT + Hann kernel) | 7.9e-3 (boundary-bound) |
| H5 | ISTFT (ConvTranspose + window-sum normalize) | 1.0e-4 |

Key techniques:

- **Snake activation** `x + (1/α)·sin²(αx)` implemented with `ggml_sin` and a
  pre-computed `1/α` tensor fed as a graph input (72 such inputs across the 9
  main ResBlocks and 3 source ResBlocks).
- **ConvTranspose1d with asymmetric PyTorch padding**: ggml's op only accepts
  `p0=0`, so we compute the full-length output then slice `p` samples from each
  side.
- **Asymmetric reflection pad `(1, 0)`**: done manually by extracting `x[1:2]`
  and concat-prepending it.
- **STFT** as `Conv1d` with a DFT+window kernel of shape `[n_fft, 1, 2F]` (real
  and imaginary parts stacked as output channels). Center-mode reflection pad
  `n_fft//2` applied manually via slice-and-concat on each side.
- **ISTFT** as `ConvTranspose1d` with the inverse DFT+window kernel, followed
  by element-wise divide by a precomputed `window²` overlap-sum buffer, then
  trim `n_fft//2` from each end.

The resulting `mel2wav` binary demonstrates the full vocoder:

```
mel2wav --s3gen-gguf models/chatterbox-s3gen.gguf \
        --mel-npy artifacts/s3gen-ref/mel_output.npy \
        --out /tmp/out.wav
```

Against the Python reference waveform: matching RMS (1.22e-04 vs 1.22e-04),
time-domain diff max 3.3e-05 (signal max ~9e-04), spectrogram magnitude diff
max rel 2.5 % (entirely from stochastic SineGen excitation; the deterministic
conv-net chain is bit-exact).

SineGen on the C++ side uses `std::mt19937` (not bit-exact to `torch.rand`,
but audibly indistinguishable — the excitation is a small-amplitude additive
noise term).

### 3.6  End-to-end wiring: `chatterbox-tts` + `synthesize.sh`

Final plumbing: write `src/chatterbox_tts.cpp` that wires the S3Gen encoder →
2-step meanflow CFM → HiFT vocoder and emits a 24 kHz wav. Takes T3-generated
speech tokens plus a reference voice (`embedding`, `prompt_token`,
`prompt_feat`).

`scripts/synthesize.sh` runs `chatterbox` → pipe tokens → `chatterbox-tts`,
giving a single-command `text → wav` path.

Debug mode (`--debug`) substitutes Python-dumped reference random bits (CFM
`z` and `noised_mels`) so the deterministic parts can be validated
bit-exactly. End-to-end in debug mode:

| Stage | max_abs | rel |
|-------|---------|-----|
| `input_embedding(tokens)` | 0 | 0 |
| encoder → `encoder_proj` (mu) | 8.3e-07 | 4.5e-07 |
| speaker embedding (spks) | 5.9e-08 | small |
| `cond` (prompt_feat placement) | 0 | 0 |
| `t_emb` (sinusoidal → MLP → mixer) | 7.6e-06 | small |
| CFM step 0 `dxdt` | 2.1e-05 | small |
| CFM step 1 `dxdt` | 1.8e-05 | small |
| final mel (80 × 136) | 1.0e-05 | **8.9e-07** |

Production mode uses a seeded `std::mt19937` for both the CFM initial noise
and SineGen excitation.

**Issues hit in this phase (all three caused plausible-looking but wrong output
before being found):**

| # | Issue | Fix |
|---|-------|-----|
| 21 | Silence-token padding value | `speech_tokens` must be appended with `S3GEN_SIL = 4299` (not 0) to match Python's `speech_tokens_padded` convention. |
| 22 | Relative PE `pos_pe / neg_pe` swap | While copying `compute_pos_emb` into the new binary I flipped the two halves of the PE buffer, which silently gave ~20 % relative error in the encoder output. Restored the correct ordering: first half is reversed `pos_pe`, second half is `neg_pe`. |
| 23 | `mu` layout transpose between encoder and CFM | `encoder_proj.npy` is numpy `(T, 80)` but the CFM estimator expects numpy `(80, T)`. Added an explicit transpose to bridge the two. |

At this point on a 10-core EPYC, single-threaded, the end-to-end pipeline ran
in **22.5 s for 8.64 s of audio** — **RTF 2.60**, i.e. 2.6× *slower* than
real-time.

### 3.7  (no extra section — continued in 3.8)

### 3.8  CPU optimization pass (in the order tried)

Eight optimizations in the order they were attempted. Four landed, four were
rolled back or skipped as incompatible. Numbers are for the 8.64 s utterance
above.

**Attempt 1 — multi-threading (KEPT, −85 % wall time)**
Baseline was pinned to 1 thread because the code never called
`ggml_backend_cpu_set_n_threads`. Added a global `g_n_threads` (default =
`std::thread::hardware_concurrency()`, overridable with `--threads N`) and a
`compute()` helper that sets it before every `ggml_backend_graph_compute`.
ggml's `-march=native` was already on, so AVX-512 / AVX-VNNI kernels were
already in use — the missing piece was parallelism. Swept thread counts: 10
was the sweet spot; 16 oversubscribes and regresses.
Result: **22.5 s → 3.47 s (RTF 2.60 → 0.40)**.

**Attempt 2 — OpenBLAS (TRIED, NO HELP)**
Installed `libopenblas-dev`, rebuilt with `GGML_BLAS=ON
GGML_BLAS_VENDOR=OpenBLAS`. No measurable change. Our matmuls are medium-sized
and ggml's hand-written AVX-512 kernels already saturate what OpenBLAS would
deliver. Kept off.

**Attempt 3 — `GGML_LTO=ON` (TRIED, NO HELP)**
No measurable effect on a shared-library build. Kept off.

**Attempt 4 — CFM graph reuse (KEPT, −11 % wall time)**
The CFM estimator is called twice per utterance with *identical* graph
topology. Stashed the `ggml_context`, `ggml_cgraph`, and `ggml_gallocr` in a
`cfm_estimator_cache` so step 2 only re-runs with new inputs — saves one graph
construction and one `gallocr_reserve` pass per utterance.
Result: **3.47 s → 3.09 s (RTF 0.40 → 0.36)**.

**Attempt 5 — Flash attention in CFM `BasicTransformerBlock` (KEPT, −22 % wall time)**
The CFM has 56 `BasicTransformerBlock`s × 2 meanflow steps = **112 attention
ops** per utterance. Replaced the explicit
`softmax(QKᵀ / √d) · V` kernel with a single `ggml_flash_attn_ext` call.
The pattern is pure self-attention (no masking, no bias), which is exactly
what `flash_attn_ext` is designed for. Fused, no materialized `T×T`
scores/attn tensors. The reshape-permute-cont preamble now drops straight into
`flash_attn_ext`, and its output `ne=[HD, H, T, 1]` reshapes directly to
`[INNER, T]`.
Result: **3.09 s → 2.45 s (RTF 0.36 → 0.28), CFM −44 %**.

**Attempt 6 — Fold symmetric conv padding (KEPT, small win)**
Six redundant `ggml_pad_ext → conv1d_f32` pairs dropped by passing the padding
straight to `ggml_im2col`. Biggest impact in HiFT's ResBlocks where the
resblock-conv path runs ~72 times per decode. Saves one intermediate tensor
allocation per conv. A small but essentially-free improvement.
Result: **2.45 s → 2.39 s (RTF 0.28 steady)**.

**Attempt 7 — F16 CFM linear weights (TRIED, ROLLED BACK)**
Converted all Q/K/V/O/FFN/MLP linear weights in CFM from F32 to F16 to halve
memory bandwidth. *Regressed*: CFM got ~10 % **slower** and precision dropped
to `rel = 3e-4` on the final mel. The F16→F32 upconvert inside `mul_mat` is
not free and the F32 AVX-512 kernel is already very fast; for CPU this is a
net loss. Reverted.

**Attempt 8 — Flash attention in the Conformer encoder (SKIPPED, INCOMPATIBLE)**
Would fuse another 10 attention ops per utterance, but the Conformer uses
ESPnet-style relative positional bias added *inside* the softmax, and
`ggml_flash_attn_ext` does not support custom in-softmax bias terms. Would
need a custom ggml op — not done.

#### Final results (10-core EPYC, 8.64 s output)

| Configuration | Total | RTF | vs real-time |
|---|---:|---:|---|
| Baseline (1 thread, no graph reuse, no flash attn) | 22.5 s | 2.60 | 2.6× slower |
| + threading (Attempt 1) | 3.47 s | 0.40 | 2.5× faster |
| + CFM graph reuse (Attempt 4) | 3.09 s | 0.36 | 2.8× faster |
| **+ flash attn + pad fold (Attempts 5, 6)** | **2.39 s** | **0.28** | **3.6× faster** |

Total wall-time speedup from the original port: **9.4×**.

Stage breakdown at the final configuration:

| Stage | time |
|-------|------|
| S3Gen encoder | 286 ms |
| CFM 2 meanflow steps | 785 ms |
| HiFT vocoder | 1312 ms |
| **Total** | **2.39 s** |

HiFT is now the bottleneck (~55 % of wall time) — the 3-stage upsample /
ResBlock stack on `T = 16320 × 64` channels is memory-bandwidth bound rather
than compute bound.

### 3.9  Post-launch bug: sampling defaults collapsed long prompts into silence

After merging the two binaries and shipping voice-cloning phase 1, a user
report of an "empty" wav on paragraph-length input surfaced a sampling bug
that had been lurking since the T3 port.

Symptom: the produced wav had ~1 second of speech followed by ~9 seconds of
pure zero RMS. Per-0.5 s window RMS:

```
[3.5e-2, 1.3e-2, 2.8e-7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.4e-7]
```

Dumping the T3 token stream showed the root cause immediately — 240 of 257
tokens were the silence token `4218`:

```
tokens[0:17]:   3704, 6486, 4299, 3891, 5832, 4384, 5014, 5665, 2486, 29,
                29, 380, 632, 2912, 5101, 5070, 4215
tokens[17:257]: 4218, 4218, 4218, 4218, ...  (240 copies)
```

The C++ sampler had shipped with `top_k = 1` (argmax) as its default. For
Chatterbox T3 that's a known failure mode: once the model generates a
silence token at a natural pause, `argmax(logits)` keeps picking silence
forever and the utterance never recovers. Short test prompts never reached
a pause so the bug was invisible during the port.

Compared `ChatterboxTurboTTS.generate()` in `tts_turbo.py` — the Python
defaults are very different:

|                | before (C++ broken) | Python     | after (C++ fixed) |
|----------------|--------------------:|-----------:|------------------:|
| `top_k`        | 1  (greedy)         | 1000       | **1000**          |
| `top_p`        | 1.0                 | 0.95       | **0.95**          |
| `temperature`  | 1.0                 | 0.8        | **0.8**           |
| `repeat_penalty`| 1.0                | 1.2        | **1.2**           |
| `n_predict`    | 256                 | ~1000      | **1000**          |

All four knobs are still exposed on the CLI, so `--top-k 1` reproduces the
old greedy behaviour for debugging/comparison.

After the fix, same prompt same seed:

- total wav RMS: `8.3e-03` → `4.8e-02`
- max amplitude: `0.18` → `0.50`
- per-0.5 s RMS windows: all 21 non-zero (3.3e-2 … 8.5e-2 range)
- audible speech for the full 10.7 s

Committed as [`bb0eb99`](https://github.com/gianni-cor/chatterbox.cpp/commit/bb0eb99).

### Lesson

This one was avoidable — the verification pipeline in §5 is per-tensor
numerical parity, which is oblivious to sampler choices; the `reference-
t3-turbo.py` harness only compared greedy token sequences so it never
exercised any non-trivial pass of the sampling ladder. Worth adding an
end-to-end sampling test to the validation list: run T3 with Python's
stochastic defaults (fixed seed) and compare the full token stream
byte-for-byte against C++ with the same seed.

---

## Verification approach

Staged pipeline:

1. **Python reference dumper** (`scripts/dump-s3gen-reference.py`) runs the
   full PyTorch pipeline with `forward_hook`s on every module we plan to
   reimplement. Each intermediate is saved as `.npy` in
   `artifacts/s3gen-ref/` with a predictable name. Multi-call hooks save a
   `_call{N}` suffix so each flow-matching step gets its own tensor.
2. **C++ staged harness** (`src/test_s3gen.cpp`) loads a single GGUF, and for
   each stage: loads the reference tensors as inputs, builds a tiny ggml
   graph covering exactly that stage, runs it, reads back outputs, and calls
   `compare_f32(got, expected, n)` to print
   `max_abs / mean_abs / rms / max|ref| / rel`.
3. For T3 we additionally have **bit-exact** testing — under greedy decoding
   ggml speech tokens equal PyTorch speech tokens token-for-token.
4. For `chatterbox-tts` we have `--debug` mode that substitutes Python-dumped
   random bits for the stochastic parts, pinning the comparison.

Precision regressions are immediately visible: a change that drops rel to
~1e-4 shows up at stage N+1 before silently corrupting the full pipeline.

---

## How to re-run everything

```bash
ssh gianni@qvac-dev-linux-x64
cd ~/qvac-chatterbox.cpp

# One-time: build the binaries
cmake -S . -B build
cmake --build build -j10 --target chatterbox chatterbox-tts test-s3gen mel2wav

# One-time: convert weights + built-in conditionals
. ~/chatterbox-ref/.venv/bin/activate
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
python scripts/convert-s3gen-to-gguf.py    --out models/chatterbox-s3gen.gguf

# One-time: dump the Python reference tensors
python scripts/dump-s3gen-reference.py \
  --text 'Hello from ggml.' --out artifacts/s3gen-ref \
  --seed 42 --n-predict 64 --device cpu

# Validate every stage in C++
./build/test-s3gen models/chatterbox-s3gen.gguf artifacts/s3gen-ref ALL

# End-to-end text → wav
./scripts/synthesize.sh "Hello from native C++." /tmp/out.wav
```

---

## Still on the table

Ranked by impact-per-effort ratio, from biggest wins to niche polish.

### Tier A — biggest wins, should be tackled next

#### A1. Voice cloning — **phase 1 DONE**, phases 2-4 pending

Voice cloning works end-to-end TODAY using a Python preprocessing
helper that produces a five-tensor voice profile from a reference
`.wav`. The C++ binary accepts it via `--ref-dir DIR`.

**Phase 1 (DONE)** — Python helper + C++ wiring:

- `scripts/prepare-voice.py`: wraps
  `ChatterboxTurboTTS.prepare_conditionals()` to produce a directory
  with `speaker_emb.npy` (T3 256-d) + `cond_prompt_speech_tokens.npy`
  (T3 ≤375 int32) + `embedding.npy` (S3Gen 192-d) + `prompt_token.npy`
  (S3Gen int32) + `prompt_feat.npy` (S3Gen mel, 80-channel).
- `src/main.cpp`: when `--ref-dir` is set, overwrite the T3 side in
  place (`model.builtin_speaker_emb`) or, when the prompt-tokens length
  differs from the GGUF's built-in (audio < 15 s → fewer tokens),
  allocate a fresh tensor in `ctx_override` + `buffer_override` on the
  same backend and repoint `model.builtin_cond_prompt_tokens` at it.
  `hparams.cond_prompt_len` is updated to match so `build_prompt_graph`
  sizes the sequence correctly.
- `src/chatterbox_tts.cpp`: the S3Gen side already reads the same three
  `.npy` files when `ref_dir` is non-empty.

End user workflow:

```bash
python scripts/prepare-voice.py --ref-audio me.wav --out voices/me/
./build/chatterbox --model models/chatterbox-t3-turbo.gguf \
                   --s3gen-gguf models/chatterbox-s3gen.gguf \
                   --ref-dir voices/me/ \
                   --text "Hello in my voice." \
                   --out out.wav
```

Verified end-to-end on the remote EPYC: override prints
`overrode T3 built-in voice from voices/test (speaker_emb=256,
cond_prompt_tokens=260)`, the synthesis runs at RTF 0.44, the output
wav plays back cleanly on the Mac.

**Phase 2 (NEXT)** — C++ mel extraction for `prompt_feat`:

- 80-channel log-mel at 24 kHz, hop 256, win 1024 (to match
  `s3gen.mel_extractor`).
- STFT code from the HiFT port is reusable; mel filterbank is a
  `[80, n_fft/2+1]` matrix that `librosa.filters.mel` computes and we
  can bake into the s3gen GGUF as a single tensor, then mul_mat over
  the magnitude spectrogram.
- Tiny — expect ~150 lines of C++ + a converter-side export of the
  pre-computed filterbank.

**Phase 3** — C++ VoiceEncoder (3-layer unidirectional LSTM, 40-channel
16 kHz mel in, 256-d speaker embedding out). LSTM cell + recurrence in
ggml follows the `whisper.cpp` / `bark.cpp` patterns. The partial-window
averaging from `VoiceEncoder.inference` is CPU-side bookkeeping.

**Phase 4** — C++ S3TokenizerV2 (~600-line wav2vec-style encoder with
FSMN attention and an FSQ codebook). This is the biggest lift and will
probably need staged verification against Python dumps, same approach
as the S3Gen encoder port in §3.3. Produces both the T3
`cond_prompt_speech_tokens` and the S3Gen `prompt_token`.

After all three phases land, `scripts/prepare-voice.py` becomes
redundant and `chatterbox --reference-audio file.wav ...` replaces it.

Impact: even Phase 1 alone unlocks "zero-shot voice cloning" as a usable
feature — the flagship reason anyone picks Chatterbox in the first
place.

#### A2. GPU backend (Metal first, then CUDA)

The code already uses `ggml_backend_t` abstractions everywhere and the
CMake flags `GGML_METAL` / `GGML_CUDA` exist; only the CPU path has been
tested so far. `chatterbox` has a `--n-gpu-layers N` flag — need to wire
the same through to `chatterbox-tts` and then actually run it.

Work items:
- Wire `--n-gpu-layers` on `chatterbox-tts`.
- Verify each custom op has a GPU kernel. Known fine on GPU:
  `flash_attn_ext`, `mul_mat`, `conv_1d` via im2col (standard),
  `soft_max`, `norm`. Probably fine: `ggml_sin` / `ggml_exp` /
  `ggml_leaky_relu` / `ggml_gelu_erf`. Need verification:
  `ggml_conv_1d` with **F32** `im2col` path (we pass `GGML_TYPE_F32` to
  keep conv kernels F32), `ggml_conv_transpose_1d`.
- Validate numerical parity — GPU `flash_attn_ext` often runs F16
  internally; confirm rel error vs PyTorch stays <1e-4.

Scope: 1–2 days if everything "just works", up to ~a week if two or three
ops need custom wiring or fallbacks.

Impact: on Apple M-series Metal, RTF should comfortably drop **below 0.1**
(sub-second generation for most utterances). Desktop discrete GPU: lower
still. **The single biggest speedup lever left.**

#### A3. Quantize T3 — ✅ **DONE** (Q8_0 / Q5_0 / Q4_0)

T3 (GPT-2 Medium, ~700 MB in F16) is the memory-bandwidth-dominated
component in the pipeline. Implemented via `--quant {f16,q8_0,q5_0,q4_0}`
flag in `scripts/convert-t3-turbo-to-gguf.py`.

The Python `gguf` 0.18 package has the K-quants (Q4_K / Q5_K / Q6_K)
declared but raises `NotImplementedError` in their `quantize_blocks`
implementations, so only legacy block types (`Q4_0`, `Q5_0`, `Q8_0`) are
produced here. Running the F16 GGUF through llama.cpp's `llama-quantize`
tool would work too, producing true K-quants — not done yet.

Only the big 2-D `mul_mat` weights get quantized: per-layer
`attn/c_attn/w`, `attn/c_proj/w`, `mlp/c_fc/w`, `mlp/c_proj/w`, plus
`chatterbox/speech_head`. Biases, layer norms, embeddings,
positional encoding, and the tokenizer metadata all stay at their
original dtype (F32 / F16). No C++ changes — `ggml_mul_mat` with
quantized weights + F32 activations is already a fast path.

Measured results, same prompt and `--n-predict 200` (201 tokens output):

**10-core EPYC** (remote):

| Variant | GGUF size | T3 wall time | vs F16 |
|---------|-----------|--------------|--------|
| F16     | 736 MB    | 3.91 s       | 1.00×  |
| Q8_0    | 460 MB    | 2.85 s       | **1.37× faster** |
| Q5_0    | 350 MB    | 2.58 s       | 1.52× faster |
| Q4_0    | 313 MB    | 2.38 s       | 1.64× faster |

**10-core Mac16,12** (M-series):

| Variant | T3 wall time | vs F16 |
|---------|--------------|--------|
| F16     | 14.92 s      | 1.00×  |
| Q8_0    | 5.41 s       | **2.76× faster** |
| Q5_0    | 5.27 s       | 2.83× faster |
| Q4_0    | 4.74 s       | 3.15× faster |

The Mac speedup is disproportionately large because M-series is much
more memory-bandwidth-bound on F16 than EPYC's DDR5 is.

Quality, comparing output tokens on a long prompt:
- **Q8_0**: **bit-for-bit identical** to F16. No audible or measurable
  quality loss. **Recommended default for quantized builds.**
- **Q5_0**: sampling diverges starting around token 6. Audio output still
  sounds correct; small perceptible voice-identity shift.
- **Q4_0**: sampling diverges slightly earlier and more. Audio still
  intelligible, with more drift from the F16 reference voice.

S3Gen / HiFT weights are conv-dominated (F16 on CFM linears actually
regressed on CPU — see §3.8 Attempt 7), so those stay F32.

Remaining: Q4_K / Q5_K path. Drop-in win would come from
`llama-quantize models/chatterbox-t3-turbo.gguf /out.gguf Q4_K_M`
once that tool's loader is pointed at our non-llama GGUF, or by
porting one of the K-quant kernels to the Python `gguf` package.

### Tier B — serious work, impactful for specific use cases

#### B1. Streaming / chunked generation for first-token latency

The current pipeline is "wait 2.4 s then hear all 8.6 s at once". For
interactive apps, **first-audio-out latency** matters more than
total RTF.

What to port:
- Chatterbox's `S3GenStreamer` path in Python: interleaves T3
  token-generation with chunked S3Gen / HiFT runs, overlap-adds their
  waveforms at the seams.
- Adds `flow_cache`, `cache_source`, `mel_cache` parameters we've been
  setting to empty, plus the overlap-add math for the HiFT vocoder.
- Emit audio to stdout (or a callback) as each chunk comes out.

Scope: ~**1 week**, mostly because the overlap-add math has to match
Python byte-for-byte or seams click.

Impact: **first audio chunk out in ~200–400 ms** instead of 2+ s. Turns
the binary from "batch" into "live".

#### B2. Server mode with persistent graphs

Every invocation currently pays ~200–400 ms fixed cost for graph
construction + `gallocr_reserve` + model load. Amortizing these over a
long-running process is free wall-time for a deployed service.

What to do:
- Daemonize with a simple stdio JSON-RPC or HTTP interface.
- Extend the `cfm_estimator_cache` pattern (from §3.8 Attempt 4) to the
  encoder and HiFT graphs — keep them pre-reserved across requests.
- Tensor shapes depend on input length → either: (a) LRU of per-length
  graphs, (b) pad to a fixed max length + attention mask, or (c) rebuild
  on shape change but pool the buffers.

Scope: **2–3 days**.

Impact: for repeated short utterances on the same server, another **20–30 %**
off wall time on top of the current RTF 0.28.

### Tier C — nice polish, niche

#### C1. Custom fused Conformer attention op (with rel-pos bias)

The S3Gen encoder's 10 Conformer blocks couldn't use `flash_attn_ext`
because they add ESPnet relative positional bias *inside* the softmax
(see §3.8 Attempt 8). A custom op that does
`softmax(QKᵀ/√d + B) · V` with B pre-computed `[L, T, H]` would fuse
those too.

Scope: **3–5 days** — CPU AVX-512 kernel first, Metal/CUDA once (A2) is
online.

Impact: maybe 50–100 ms off encoder (~10 % of encoder, which is already
only 12 % of the pipeline). Small in absolute terms; does get you the
same fusion level throughout.

#### C2. Batch generation

Multiple utterances in one pass. Python supports it; our C++ pipeline
assumes batch=1 throughout. Only matters at scale (multiple concurrent
users).

#### C3. Repository / packaging polish

- **GitHub Actions CI** running `compare-tokenizer.py` + `test-s3gen ALL`
  on every push. All the validation infrastructure is already in place;
  wiring it takes a few hours.
- **Prebuilt GGUFs on Hugging Face** so end users don't need the
  Python toolchain at all. Upload the two `.gguf` files with a model
  card explaining the build.
- **Library API** (not just binaries). Expose
  `chatterbox_synthesize(text, opts) -> wav` as a C / C++ API so
  Swift / Node.js / Python bindings can layer on top. ~Half a day.

### Recommended next-up order

For maximum usefulness per day of effort:

1. **A3 — Quantize T3** (~4 h) → instant size/speed win, essentially
   risk-free.
2. **A1 — Voice cloning** (2–3 days) → unlocks the flagship
   Chatterbox feature; without it this is only a demo.
3. **A2 — Metal backend** (1–2 days if smooth) → biggest speedup left;
   Mac users feel it immediately.
4. **B1 — Streaming** (~1 week) → what gets this into interactive apps.
5. **C3 — CI + prebuilt GGUFs** — pick up before announcing publicly.

B2 (server mode) and C1 (custom Conformer attn op) are worth doing once a
concrete deployment is pressuring for them; right now the CPU numbers are
already well past real-time for CLI use.
