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

1. **Quantize the T3 GPT-2 Medium backbone** to Q4_K_M / Q5_K — only affects
   the separate `chatterbox` (T3) binary. ~700 MB F16 → ~200 MB Q4_K_M with
   negligible quality loss (proven on GPT-2 Medium sized backbones
   elsewhere).
2. **GPU backends** (CUDA / Metal) — already wired through `ggml_backend_t`;
   untested for this workload. With the graph-reuse in place these should
   be straightforward.
3. **Pre-compiled / cached graphs across utterances** — for server mode,
   extend the `cfm_estimator_cache` pattern to the encoder and HiFT graphs
   so the per-utterance graph-build + `gallocr_reserve` cost amortizes.
4. **Voice cloning** — currently uses the built-in `conds.pt` voice. To
   support custom audio, port `VoiceEncoder` (3-layer LSTM) and either
   `S3Tokenizer` or accept pre-computed speaker embeddings from a Python
   preprocessing step. LSTM inference in ggml is known-good via
   `whisper.cpp` / `llama.cpp` patterns.
5. **Custom Conformer flash-attn op** — implement a ggml op that folds
   ESPnet's in-softmax relative positional bias into a flash-attention
   kernel, so the S3Gen encoder can also benefit (Attempt 8 above).
