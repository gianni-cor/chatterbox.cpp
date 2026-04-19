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

**Speed** (10 s sentence, seed 42, `gen_RTF = (T3_INFER + S3GEN_INFER) / audio_ms`):

| Backend                     | `gen_RTF` | Wall  | vs ONNX addon |
|-----------------------------|----------:|------:|--------------:|
| CPU (10-core EPYC, F16)     | 0.70      | 8.2 s | 3.6× faster   |
| **Vulkan (RTX 5090, Q4_0)** | **0.06**  | **1.8 s** | **7.8×** |
| **Metal (M3 Ultra, Q4_0)**  | **0.13**  | **1.9 s** | **7.4×** |
| ONNX q4 addon (CPU baseline) | 1.06     | 13.9 s | 1.0×         |

GPU support and Metal kernel fixes are described in §3.11 / §3.12;
the layout-friendly KV cache + Flash Attention pass that produced the
numbers in this table is in §3.13.

---

## Repository layout

```
qvac-chatterbox.cpp/
  ggml/                           vendored ggml checkout (see patches/)
  patches/
    ggml-metal-chatterbox-ops.patch   Metal op fixes: diag_mask_inf, pad_ext,
                                      faster conv_transpose_1d (applied to ggml/
                                      during setup; see patches/README.md)
    README.md                         why each patch exists + how to drop it
  src/
    main.cpp                      T3 runtime + unified CLI (chatterbox binary)
    chatterbox_tts.cpp            S3Gen encoder + CFM + HiFT (reusable entry)
    gpt2_bpe.{h,cpp}              self-contained GPT-2 byte-level BPE tokenizer
    voice_features.{h,cpp}        wav I/O, resample, mel, fbank, LUFS
    voice_encoder.{h,cpp}         VoiceEncoder 256-d speaker embedding
    campplus.{h,cpp}              CAMPPlus 192-d speaker embedding
    s3tokenizer.{h,cpp}           S3TokenizerV2 (wav → S3 speech tokens)
    test_s3gen.cpp                staged verification harness (stages A..H5)
    test_metal_ops.cpp            parity test for the patched Metal kernels
    mel2wav.cpp                   mel → wav demo binary (HiFT only)
    npy.h                         minimal .npy loader + compare helpers
  scripts/
    convert-t3-turbo-to-gguf.py       T3 weights + tokenizer + VE + builtin voice → GGUF
    convert-s3gen-to-gguf.py          S3Gen encoder + CFM + HiFT + CAMPPlus
                                      + S3TokenizerV2 + mel filterbanks → GGUF
    dump-s3gen-reference.py           PyTorch → .npy intermediates for test-s3gen
    dump-campplus-reference.py        PyTorch → .npy intermediates for test-campplus
    dump-s3tokenizer-reference.py     PyTorch → .npy intermediates for test-s3tokenizer
    reference-t3-turbo.py             PyTorch T3 + compare against C++
    compare-tokenizer.py              10-case tokenizer comparison against HF
    synthesize.sh                     text → wav wrapper (chatterbox binary)
  models/
    chatterbox-t3-turbo.gguf      T3 + tokenizer conditionals
    chatterbox-s3gen.gguf         flow + mel2wav weights + built-in voice
    t3-{q8_0,q5_0,q4_0}.gguf      quantized T3 variants (A3)
  CMakeLists.txt                  top-level: add_subdirectory(ggml) + targets
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

### 3.10  Benchmark: chatterbox.cpp vs ONNX addon on the same machine

Compared end-to-end throughput against the in-house
`qvac-lib-infer-onnx-tts` addon (ONNX Runtime backend, pre-built q4
Chatterbox models at 692 MB on disk). Same 10-core EPYC host, same
prompt ("Hello from native C plus plus. This audio was generated end
to end on CPU using ggml."), built-in voice on both sides, `--threads
10` for ggml, ORT's own default threading for ONNX. Instrumented the
ggml binary with explicit `T3_LOAD_MS` / `T3_INFER_MS` /
`S3GEN_LOAD_MS` / `S3GEN_INFER_MS` markers so load and generate
phases can be split cleanly. Each configuration run three times after
a disk-cache warm-up.

**Model footprint on disk:**

| | Size |
|---|---:|
| ONNX q4 (5 files) | 692 MB |
| ggml F16 (T3 + S3Gen) | 1285 MB |
| ggml Q8_0 (T3 + S3Gen) | 1004 MB |
| ggml Q5_0 (T3 + S3Gen) | 893  MB |
| ggml Q4_0 (T3 + S3Gen) | 857  MB |

**Per-stage wall-clock (median of 3 runs, milliseconds):**

| Pipeline      | T3 load | T3 gen | S3Gen load | S3Gen gen | Audio | **Total** | RTF (total) |
|---------------|---:|---:|---:|---:|---:|---:|---:|
| **ggml Q4_0** | **213** | 1790 | 366 | 1998 | 6480 | **4455** | **0.69** |
| ggml Q5_0     |  231 | 1966 | 353 | 2002 | 6640 |  4641 | 0.70 |
| ggml Q8_0     |  305 | 2047 | 370 | 2001 | 6560 |  4823 | 0.73 |
| ggml F16      |  468 | 2691 | 364 | 1928 | 6560 |  5562 | 0.85 |
| **ONNX q4**   |  ~4250 (4 files, serialized) | — | — | ~6830 | 5880 | **11050** | **1.88** |

(ONNX Runtime's backend doesn't expose a comparable per-sub-model
breakdown, so its `load` is the wall-clock time from `model.load()`
calling through ORT init across all four `.onnx` files, and `gen` is
the time the single `model.run()` call takes.)

**Aggregated: load vs. generate, load+gen together:**

| Pipeline      | **Load** | **Generate** | **Total wall** | **RTF (total)** |
|---------------|---:|---:|---:|---:|
| **ggml Q4_0** |  **579 ms** |  **3788 ms** |  **4455 ms** |  **0.69**  |
| ggml Q5_0     |  584 ms |  3968 ms |  4641 ms |  0.70 |
| ggml Q8_0     |  675 ms |  4048 ms |  4823 ms |  0.73 |
| ggml F16      |  832 ms |  4619 ms |  5562 ms |  0.85 |
| **ONNX q4**   | **4250 ms** | **6830 ms** | **11050 ms** | **1.88** |

**Headline numbers** (best ggml variant vs ONNX):

- **Load: ggml Q4_0 is 7.3× faster** — 579 ms vs 4250 ms. The four
  ONNX files initialise serially and each one does its own tensor
  plumbing; ggml mmaps the two GGUFs and rebinds through the unified
  backend buffer in ~half a second total.
- **Generate: ggml Q4_0 is 1.8× faster** — 3788 ms vs 6830 ms.
- **Total (load + generate): ggml Q4_0 is 2.48× faster** —
  4.46 s vs 11.05 s.
- Even **ggml F16 beats ONNX q4** on total wall (5.56 s vs 11.05 s,
  1.99× faster) *despite carrying 2× the weights* — the ONNX backend
  loses to an un-quantized ggml build on the same CPU.
- **RTF < 1** (faster than real-time) happens on every ggml variant
  tested; ONNX stays at 1.88× real-time for this prompt.

Numbers are for a ~6 s utterance; the ggml pipeline's ~2 s of fixed
S3Gen+HiFT cost amortizes better on longer input, so the gap widens
in ggml's favour as prompt length grows.

### 3.11  Vulkan + Metal backends

CPU performance was already past real-time, but a lot of the T3 and
CFM work is embarrassingly parallel, so enabling the GGML GPU backends
was the obvious next step. Touched three files:

- `CMakeLists.txt` — added a `GGML_VULKAN` propagation block mirroring
  the existing `GGML_CUDA` / `GGML_METAL` ones.
- `src/main.cpp` — extended `init_backend(n_gpu_layers)` with a
  `ggml_backend_vk_init(0)` path guarded by `#ifdef GGML_USE_VULKAN`.
  CUDA / Metal paths were already there.
- `src/chatterbox_tts.cpp` — added a symmetric `s3gen_init_backend`
  so the S3Gen side honours the same `--n-gpu-layers` flag, plus a
  new `n_gpu_layers` field on `s3gen_synthesize_opts`.

Two op-level changes in our code were required *because* Metal's
dispatcher didn't have those ops (the actual Metal kernel fixes land
in §3.12):

1. **T3 attention**: `ggml_soft_max(ggml_diag_mask_inf(ggml_scale(KQ,
   s), n_past))` → `ggml_soft_max_ext(KQ, mask, s, 0.0f)` with an
   explicit `[n_kv, N]` causal mask tensor uploaded from
   `eval_prompt`. The step path (N=1) passes a null mask. No-op for
   CPU / Vulkan; necessary for Metal.
2. **S3Gen zero padding**: 6 call sites used `ggml_pad_ext` with
   non-zero front padding. Added a `zero_pad_dim0(ctx, x, p_front,
   p_back)` helper that expresses the same semantics via
   `concat(scale(view, 0.0f), x)` so it runs on every backend with
   well-defined zeros.

First result on the Linux remote (RTX 5090 + Vulkan), same 10 s
sentence as §3.10:

| Variant       | T3 load | T3 gen | S3Gen load | S3Gen gen | Audio  | `gen_RTF` | Wall  |
|---------------|--------:|-------:|-----------:|----------:|-------:|----------:|------:|
| Vulkan F16    |  562 ms |  600 ms |  490 ms    | 279 ms    | 10.5 s | **0.08** | 2.10 s |
| Vulkan Q8_0   |  450 ms |  557 ms |  472 ms    | 272 ms    | 10.6 s | 0.08     | 1.91 s |
| Vulkan Q5_0   |  348 ms |  562 ms |  470 ms    | 276 ms    | 10.9 s | 0.08     | 1.82 s |
| Vulkan Q4_0   |  331 ms |  522 ms |  493 ms    | 275 ms    | 10.3 s | 0.08     | 1.78 s |

Quantization makes T3 load noticeably smaller but barely moves
inference — T3 is autoregressive (one token at a time on a 5090 has
plenty of spare lanes) and S3Gen is already short. End-to-end goes
from 8.17 s (CPU F16) → **1.78 s** (Vulkan Q4), for the same 10 s of
audio. `gen_RTF = 0.08` = 13× real-time.

On the M3 Ultra Metal side, things didn't fly immediately: T3 aborted
on the first attention layer with `unsupported op 'DIAG_MASK_INF'`,
then S3Gen aborted with `unsupported op 'PAD'`. Once those two
op-level workarounds above were in place, HiFT decode was completing
but taking **~15 s for 1.2 s of audio** — Metal's
`conv_transpose_1d` kernel is pathological for HiFT-sized inputs.

Pragmatic interim fix: when the main backend is Metal, load a second
CPU copy of the S3Gen GGUF and route `run_f0_predictor`,
`run_stft`, and `run_hift_decode` through it. Encoder + CFM still run
on Metal. Costs ~1 GB extra RAM but brings Metal `gen_RTF` to ~0.25.
That's what committed as `795963a` ("backend: enable Vulkan + Metal
for T3 and S3Gen").

### 3.12  ggml-metal kernel patches

To get rid of the CPU fallback for HiFT and close the gap with
Vulkan, patched `ggml/src/ggml-metal/` itself. The patch is shipped
as `patches/ggml-metal-chatterbox-ops.patch` (based on upstream
`58c3805`, `sync : llama.cpp`); the main README instructs a fresh
clone to `git apply` it after cloning ggml.

A new `test-metal-ops` binary runs each patched kernel against the
CPU reference at HiFT-realistic shapes. All cases pass with
`max_abs ≤ 1.5e-6`.

**Patch 1 — `DIAG_MASK_INF` on Metal** (was: op simply absent from
the dispatcher):

- New `kernel_diag_mask_inf_f32` — ports the CUDA formulation
  (`dst[i] = src[i] - (col > n_past + row % rows_per_channel) *
  FLT_MAX`) so downstream softmax yields proper zeros.
- New `ggml_metal_kargs_diag_mask_inf`, library pipeline getter,
  op encoder, dispatcher case, and `supports_op` entry.

**Patch 2 — `PAD` with front padding** (was: kernel ignored
`op_params[0,2,4,6]` which is where `ggml_pad_ext` stores the front
amounts; `supports_op` hard-rejected any non-zero front pad):

- Extended `ggml_metal_kargs_pad` with `lp0..lp3`.
- Rewrote `kernel_pad_f32` to translate each output coord by
  `i0x = i0 - lp0` etc., and write `0.0` outside `[0, ne00)`.
- Relaxed `supports_op` to `src0->type == F32 && dst->type == F32`.

**Patch 3 — `CONV_TRANSPOSE_1D` speedup** (was: ~100× slower than
CPU on HiFT-sized inputs):

The old kernel was scalar — one thread per output pixel, iterating
over the full `IC × IL` inputs inside a branch `if (ol >= i*s0 && ol
< i*s0 + K)`. Two orthogonal fixes:

1. **Tighten the input-position loop** to only the `i`s that actually
   contribute. For fixed `ol`, valid `i` is
   `[max(0, ⌈(ol - K + 1)/s0⌉), min(IL-1, ol/s0)]` — at most
   `K/s0 + 1` iterations. On ups[0] (s0=8, K=16, IL≈130) this
   collapses the inner loop from 130 iterations → 3.
2. **Parallelise `IC` across a 32-thread simdgroup** and reduce with
   `simd_sum`. Host-side dispatch widens from 1 thread per
   threadgroup → 32 (one simdgroup).

Measured on M3 Ultra, HiFT decode (part of a 10 s sentence):

```
  hift_decode: 15021 ms → 350 ms          (≈ 40× speedup)
  gen_RTF   :   0.25  → 0.18              (CPU-fallback removed)
  wall      :   3.36 s → 2.51 s
```

With the patch applied and the CPU-fallback for HiFT removed,
end-to-end on the M3 Ultra for the same 10 s sentence, seed 42,
averaged over 3 runs:

| Variant       | T3 load | T3 gen | S3Gen load | S3Gen gen | `gen_RTF` | Wall  |
|---------------|--------:|-------:|-----------:|----------:|----------:|------:|
| Metal F16     |  280 ms | 1326 ms |  295 ms    | 577 ms    | **0.19** | 2.51 s |
| Metal Q8_0    |  216 ms | 1330 ms |  302 ms    | 598 ms    | 0.18     | 2.48 s |
| Metal Q5_0    |  186 ms | 1393 ms |  293 ms    | 611 ms    | 0.19     | 2.51 s |
| **Metal Q4_0**|  **175 ms** | **1274 ms** | **295 ms** | **594 ms** | **0.18** | **2.36 s** |

Autoregressive T3 now dominates wall time (`T3_INFER` ≈ 1.3 s of
~260 tokens at one-token-at-a-time on a 60-core Apple GPU) — that's
the next thing to chip away at. On the 5090 the same token stream
runs in ~0.55 s because the shader count is ~360× higher.

Committed as `894c4b1` ("metal: patch ggml to fix diag_mask_inf,
pad_ext, conv_transpose_1d"). `i`m not a fan of forking ggml just
for this, so the patch is tiny and easy to drop once upstream picks
up equivalent fixes; see `patches/README.md` for what to do in that
case.

### 3.13  T3 Flash Attention with a layout-friendly KV cache

After §3.11 / §3.12 the dominant wall-clock cost in Chatterbox became
T3's autoregressive step (≈ 1.3 s of a ~2.4 s run on Metal M3 Ultra
Q4_0).  An earlier attempt to swap the explicit
`soft_max_ext(mul_mat(K,Q), mask) + mul_mat(V_trans)` chain for
`ggml_flash_attn_ext` ran into a deal-breaker: the KV cache was laid
out `[HD, n_head, n_ctx]` per layer but `flash_attn_ext` wants
`[HD, n_ctx, n_head]`.  Every step had to `ggml_cont(ggml_permute(K))`
over a tensor that grew with `n_past`, and the extra kernel dispatches
wiped out FA's savings.

Fix: store the cache the way FA reads it.

- Same total size per layer (`HD * n_ctx * n_head` == `n_embd * n_ctx`),
  so no allocation changes.
- Write path (step or prompt): Kcur / Vcur are viewed as
  `[HD, n_head, N]`, permuted to `[HD, N, n_head]`, then one
  `ggml_cpy` per tensor into a strided cache view at
  `[HD, n_past:n_past+N, n_head]`.  For the step path N=1 the permute
  is a no-op in memory.
- Read path: `ggml_view_3d(memory_k, HD, L, n_head, nb=[4, HD*4,
  HD*n_ctx*4], offset=il*layer_size)` is exactly the shape FA needs,
  with no `permute + cont`.
- Mask: switched from F32 to F16 (ggml FA requires F16 on Metal;
  other backends accept it too).  N=1 path passes `nullptr` since
  every KV position is in the past.

Measured on M3 Ultra, same 10 s sentence, seed 42, `--threads 20`,
`--n-gpu-layers 99`, averaged over 3 warm runs:

| Variant  | T3 infer before | T3 infer after | Δ     | Wall before | Wall after | `gen_RTF` |
|----------|----------------:|---------------:|------:|------------:|-----------:|----------:|
| F16      |          1372 ms|         983 ms | −28 % |      2.51 s |     2.15 s | 0.189 → 0.157 |
| Q8_0     |          1371 ms|         985 ms | −28 % |      2.48 s |     2.12 s | 0.182 → 0.149 |
| Q5_0     |          1445 ms|        1063 ms | −26 % |      2.51 s |     2.18 s | 0.186 → 0.152 |
| **Q4_0** |      **1274 ms**|     **965 ms** | **−24 %** | **2.36 s** | **2.06 s** | **0.176 → 0.144** |

And the same change on Vulkan 5090 (Linux remote):

| Variant  | T3 infer before | T3 infer after | Δ     |
|----------|----------------:|---------------:|------:|
| F16      |           600 ms|         410 ms | −32 % |
| Q4_0     |           522 ms|         356 ms | −32 % |

So the new layout is not just a Metal-shaped win — it speeds up every
GPU backend, because the previous `permute + cont` per layer per step
was cheap on NVIDIA too but not free.  CPU builds see a similar graph
shape (fewer intermediate nodes) and stay neutral.

Output sampling is *not* bit-exact against the old path: FA runs its
own internal reductions in different order and the mask lives in F16
instead of F32, so token counts can shift by ±2 % (e.g. F16 went from
248 → 244 tokens on the bench prompt).  Audio remains perceptually
identical; this is the same kind of drift that moving to FA causes
anywhere else in ggml.

Committed as part of the Metal optimization sequence alongside the
earlier `patches/ggml-metal-chatterbox-ops.patch`.

### 3.14  Zero-cont Q view via strided QKV access

After §3.13, each T3 attention layer still did two `ggml_cont`s on Q
per step: one `cont_3d` to densify the strided view of `Qcur`, and an
outer `cont` after the head-permute.  Both turn into
`kernel_cpy_f32_f32` dispatches on Metal.

Observation: the entire QKV output `cur` is already contiguous.  Q,
K, and V are just fixed byte offsets into the same tensor (0,
`n_embd * 4`, `2 * n_embd * 4` respectively).  With Metal's
`flash_attn_ext` accepting non-contiguous Q via explicit strides (the
same flexibility I used for K/V in §3.13), I can drop both conts and
express Q directly as a `ggml_view_3d` with layout `[HD, N, n_head]`:

```
nb0 = 4, nb1 = 3 * n_embd * sizeof(float), nb2 = HD * sizeof(float)
```

Same trick for the Kcur/Vcur sources that go into the KV-cache write
path — one view each, no permute + cont pair.

Removes 24 kernel dispatches per step (`cont` × 24 layers); since T3
step time on Metal is almost entirely dispatch-bound at ~9 µs each,
this shows up straight in the numbers.

Measured on M3 Ultra (same 10 s sentence, seed 42, 3-run warm average):

| Variant  | T3 infer §3.13 | T3 infer §3.14 | Δ     | Wall §3.13 | Wall §3.14 |
|----------|---------------:|---------------:|------:|-----------:|-----------:|
| F16      |         983 ms |         909 ms | −7.5% |    2.15 s  |   **2.08 s** |
| Q8_0     |         985 ms |         906 ms | −8.0% |    2.12 s  |   **2.03 s** |
| Q5_0     |        1063 ms |         984 ms | −7.4% |    2.18 s  |   **2.09 s** |
| **Q4_0** |     **965 ms** |     **886 ms** | **−8.2%** | **2.06 s** | **1.98 s** |

Vulkan RTX 5090 sees <3 % change in T3 infer — dispatch overhead is
much smaller there relative to the actual compute, so there's less to
save.  No regression on Vulkan, and the code simplifies.  CPU stays
neutral (same graph topology, fewer intermediate nodes).

Sampling output is not bit-exact against §3.13 either — same reason as
before, FA reductions are sensitive to operand stride.  Token counts
shift within ±1 % at the same seed.

### 3.15  ggml-metal: fuse `mul_mat + add(bias)` for Q-variant matvec

Even after §3.14 the T3 step path still dispatched two Metal kernels
per linear layer — `mul_mv` for the matmul itself, then `bin_fuse` for
the following `add(bias)`.  T3 has 4 such linears per layer
(QKV proj, attn proj, MLP fc, MLP proj) × 24 layers = 96 extra bias
kernels per step.  At ~9 µs dispatch overhead on M3 Ultra that's
~900 µs/step / ~240 ms over a 260-token generation.

Patched ggml-metal to fuse these directly inside the mul_mv kernel
(third addition to `patches/ggml-metal-chatterbox-ops.patch`):

1. New function constant `FC_mul_mv_has_bias` at `FC_MUL_MV + 2`.
2. Each Q-variant top-level kernel (`kernel_mul_mv_q4_0_f32`,
   `_q4_1_f32`, `_q5_0_f32`, `_q5_1_f32`, `_q8_0_f32`) picks up an
   extra `device const char * bias` buffer argument and calls a tiny
   `helper_mv_add_bias<NR0>` immediately after the existing impl.
   The post-pass only runs when the function constant is true and
   only one thread per row does the add (no cross-threadgroup
   synchronisation needed; each threadgroup writes and then reads
   back only its own output rows).
3. `ggml_metal_op_mul_mat` gets a `ctx->use_fusion &&
   kernel_supports_bias` look-ahead: if the next op is an `ADD` with
   a contiguous F32 `[ne0, 1]` bias, we compile the pipeline with
   `has_bias=true`, bind the bias buffer to slot 4, redirect the
   matmul's `dst` to the ADD's output tensor, and return `n_fuse=2`
   so the dispatcher skips the ADD.  The shared pipeline name
   (`…_bias=1`) makes the fused variant cache-coherent with the
   non-fused one.
4. For kernels not yet wired (F16/BF16 `mul_mv_t_t`, the `_4` SIMD
   variants, all the K-quants and IQ variants) the fusion is
   suppressed by `kernel_supports_bias`, the pipeline compiles with
   `has_bias=false`, and the kernel's `if (FC_mul_mv_has_bias)` is
   dead-code eliminated.  MoE `mul_mv_id` keeps calling the original
   impl via `mmv_fn` unchanged; the impl signature itself was not
   touched.

Measured on M3 Ultra, 10 s sentence, seed 42, 3-run warm average:

| Variant  | T3 before §3.15 | T3 after §3.15 | Δ      | Wall before | Wall after |
|----------|----------------:|---------------:|-------:|------------:|-----------:|
| F16      |          909 ms |         915 ms | ~flat  |   2.08 s    |  2.26 s    |
| Q8_0     |          906 ms |         819 ms | −9.6%  |   2.03 s    |  2.02 s    |
| Q5_0     |          984 ms |         840 ms | −14.6% |   2.09 s    |  1.96 s    |
| **Q4_0** |      **886 ms** |     **766 ms** | **−13.5%** | **1.98 s**  | **1.87 s** |

F16 is flat because the kernel it hits (`mul_mv_f16_f32_4`) isn't in
the supported list yet; extending to those variants is a mechanical
follow-up (touches `helper_mv_reduce_and_write` + the 3 `_t_t` /
`_t_t_4` / `_t_t_short` templates in the same way).

Vulkan RTX 5090 unchanged (347 → 343 ms on Q4_0 — noise).  CPU
unaffected (Metal-only change).

Total Metal Q4_0 journey (pre-FA → end of §3.15):

```
              T3 infer   Wall    gen_RTF
pre-FA         1274 ms   2.36 s   0.176
§3.13 FA+KV     965 ms   2.06 s   0.144     -24%
§3.14 Q views   886 ms   1.98 s   0.131     -30%
§3.15 bias fn   766 ms   1.87 s   0.119     -40%
```

**40 % faster T3 inference, 21 % faster end-to-end wall** than the
pre-optimization baseline on the same M3 Ultra — all via Metal
kernel + graph-shape changes, no model changes.

### 3.16  Metal: extend mat-vec fusion to `MUL_MAT + ADD + ADD`; Vulkan/CPU already optimal

While investigating whether the §3.15 fusion could also apply to
Vulkan and CPU, two findings:

- **Vulkan already has it.** `ggml_vk_can_fuse` in upstream recognises
  `MUL_MAT + ADD` *and* `MUL_MAT + ADD + ADD`, and the mat-vec shaders
  (`vulkan-shaders/mul_mat_vec_iface.glsl`) have dedicated `Fuse0` /
  `Fuse1` buffer bindings for the two optional adds.  Running
  `GGML_VK_DISABLE_FUSION=1` on the 5090 pushes T3 Q4_0 from 346 →
  413 ms (3-run avg), a real 16 % speedup that was silently helping us
  before.  Nothing to add on Vulkan.
- **CPU has no op-level fusion framework.**  But it also has ~zero
  per-op dispatch overhead (ggml-cpu just calls the next op's compute
  function directly), and the matmul output stays in L1 cache
  (`n_embd=1024` × 4 B = 4 KB) so the intermediate round-trip is
  essentially free.  Estimated gain from fusion: < 1 %.  Not worth the
  plumbing work.

That left Metal, where §3.15 covered `MUL_MAT + ADD(bias)` but not the
3-op form `MUL_MAT + ADD(bias) + ADD(residual)` used by T3's attn-proj
and MLP-proj linears.  Extended the Metal patch to match Vulkan's
fusion surface:

- New function constant `FC_mul_mv_has_residual` at `FC_MUL_MV + 3`.
- Each Q-variant top-level kernel gains a second buffer binding
  (`device const char * residual` at slot 5).  `helper_mv_add_bias`
  now applies both the bias broadcast and the per-element residual
  add; both branches are gated on their respective function constants
  so non-fused call sites specialise them away.
- `ggml_metal_op_mul_mat` tries `{MUL_MAT, ADD, ADD}` first (requires
  bias-shaped src1 on ADD1 and full-shape F32-contiguous on ADD2),
  falls back to `{MUL_MAT, ADD}` from §3.15.  Returns `n_fuse=3` /
  `n_fuse=2` accordingly.
- Pipeline names now carry `_bias=?_res=?` so fused/non-fused variants
  are cached independently by the library.

**Correctness bug caught while writing the 3-op variant.**  §3.15's
helper had `if (tiisg != 0 || sgitg != 0) return;`, so only simdgroup
0 added bias.  That's correct for Q8_0 (all simdgroups cooperate on
the same `r0`) but **wrong for Q4/Q5** where each simdgroup writes
its own `r0 = (tgpig.x*NSG + sgitg)*NR0`, silently dropping bias from
the rows computed by simdgroups ≥ 1.  Output was "close enough" to
sound right but not numerically correct.  Fixed by moving the
`sgitg` gate to the callers: Q-n kernels call the helper from every
simdgroup with their own `r0`; Q8_0 wraps the call in
`if (sgitg == 0)`.  Token counts snapped back to the pre-fusion
trajectory once this was right.

Measured on M3 Ultra, 10 s sentence, seed 42, 3-run warm average:

| Variant  | T3 before §3.16 | T3 after §3.16 | Δ     | Wall before | Wall after |
|----------|----------------:|---------------:|------:|------------:|-----------:|
| F16      |          915 ms |         913 ms | flat  |   2.26 s    |  2.27 s    | (fusion N/A)
| **Q8_0** |      **819 ms** |     **794 ms** | **−3 %** | **2.02 s** | **1.94 s** |
| Q5_0     |          840 ms |         873 ms | +4 %  |   1.96 s    |  2.01 s    | (tokens +15)
| Q4_0     |          766 ms |         770 ms | flat  |   1.87 s    |  1.88 s    | (tokens +6)

Smaller than the headline "save 48 dispatches × 9 µs" estimate
suggested, because Metal's scheduler overlaps consecutive small
dispatches — the `bin_fuse` the fused kernel replaces was already
running concurrently with later work.  Q8_0 still sees a clean 3 %
win; Q4/Q5 are noise after accounting for token-count drift.  Still
worth committing: matches Vulkan's fusion surface, fixes the latent
§3.15 bias correctness bug, and closes the last dispatch-per-linear
gap vs Vulkan.

### Cross-backend summary

Same 10 s sentence, seed 42, `gen_RTF` is inference-only (excludes
load time):

| Backend (weights)             | T3 gen | S3Gen gen | `gen_RTF` | Wall  | Real-time mult |
|-------------------------------|-------:|----------:|----------:|------:|---------------:|
| CPU Linux (F16, 8 threads)    | 3998 ms | 2905 ms   | 0.70      | 8.17 s | 1.4×          |
| Vulkan 5090 (F16)             |  402 ms |  282 ms   | 0.064     | —       | 15.6×         |
| Vulkan 5090 (Q4_0)            |  347 ms |  284 ms   | 0.058     | —       | 17.1×         |
| Metal M3 Ultra (F16)          |  915 ms |  567 ms   | 0.150     | 2.26 s | 6.7×          |
| Metal M3 Ultra (Q4_0)         |  766 ms |  596 ms   | 0.128     | 1.87 s | 7.8×          |
| ONNX q4 addon (CPU, Linux)    |     — (not exposed) |     — | 1.06      | 13.91 s | 0.94×        |

The ONNX addon is shown as a baseline because it's what
`qvac-lib-infer-onnx-tts` ships today. Every ggml configuration —
including CPU F16 on the same host — beats it.

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

#### A1. Voice cloning — **ALL PHASES DONE** (pure C++ voice cloning, no Python at runtime)

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

**Phase 2a (DONE)** — C++ WAV I/O + sinc resampler + 80-ch log-mel at 24 kHz:

- `src/dr_wav.h` (public-domain single header, MIT-0 fallback) vendored
  as a bundled WAV loader (all PCM variants, any sample rate,
  auto-mono).
- `src/voice_features.{h,cpp}`: `wav_load`, `resample_sinc`
  (Kaiser-windowed, beta=8.6, configurable tap count), and
  `mel_extract_24k_80`. The mel extractor is a direct port of
  `s3gen.utils.mel.mel_spectrogram` (`n_fft=1920`, `hop=480`,
  `win=1920`, `fmin=0`, `fmax=8000`, `center=False`, reflect-pad 720).
- `scripts/convert-s3gen-to-gguf.py` now also bakes in the precomputed
  librosa mel filterbank (`librosa.filters.mel(sr=24000, n_fft=1920,
  n_mels=80, fmin=0, fmax=8000)`, a `(80, 961)` float32 matrix) as
  `s3gen/mel_fb/24k_80`. Runtime has no librosa dep.
- Two validation binaries: `test-resample` (24 kHz → 48 kHz → 24 kHz
  round-trip on a 4-tone signal, expects **> 60 dB SNR**) and
  `test-voice-features MODEL.gguf REF.wav PROMPT_FEAT.npy` (compares
  C++ 80-ch log-mel against a Python-dumped `prompt_feat.npy`).

Measured on 10-core EPYC:

| Check                                            | Result |
|---|---|
| Resampler round-trip (4-tone, 24k ↔ 48k)         | **95.75 dB SNR** |
| Mel parity vs Python `prompt_feat.npy` (rel)     | **8.3e-08**      |

(The ~500-frame Python reference truncates at DEC_COND_LEN = 10 s; the
C++ side produces an extra ~20 frames for a 10.4 s input wav but the
overlapping 500 × 80 values match to float precision.)

Implementation notes:

- First attempt at `resample_sinc` was a polyphase decomposition with
  a Kaiser-windowed sinc prototype; the phase-indexing convention was
  subtly wrong and gave **0 dB SNR** on the round-trip. Swapped for
  straightforward "fractional-index sinc interpolation at each output
  sample" which is correct and still fast enough for one-shot voice
  preprocessing.
- `mel_extract_24k_80` uses a naive O(n_fft) DFT per frame, not an FFT.
  For a 10 s reference that's ~520 frames × 1920 × 961 ≈ 960 M mults,
  well under 2 s on CPU. Fine for preprocessing; an FFT is a trivial
  follow-up if this ever needs to be streaming.

**Phase 2b (DONE)** — `--reference-audio PATH.wav` wired into `main.cpp`.
The CLI now accepts a reference wav, runs the whole WAV→prompt_feat
chain in C++, and injects the result into `s3gen_synthesize_opts`
(new `prompt_feat_override` field) so the S3Gen+HiFT pipeline consumes
it directly — no temp file, no npy round-trip. The other four voice
tensors still come from `--ref-dir` for now.

User workflow:

```bash
python scripts/prepare-voice.py --ref-audio me.wav --out voices/me/
./build/chatterbox \
    --model models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf models/chatterbox-s3gen.gguf \
    --ref-dir voices/me/ \
    --reference-audio me.wav \
    --text "Voice-cloned with C++ mel." \
    --out out.wav
```

Verified end-to-end: `voice: prompt_feat shape=(520, 80)` /
`prompt_feat: using C++ override (520 mel frames)` / audible cloned
voice at RTF 0.76 on 10-core EPYC.

**Phase 2c (DONE)** — C++ VoiceEncoder: 3-layer unidirectional LSTM +
Linear(256 → 256) + ReLU + L2-normalise, 40-channel 16 kHz power-mel in,
256-d speaker embedding out.

New files:
- `src/voice_encoder.{h,cpp}` — weights loader (reads 14 tensors from
  the t3 GGUF + `voice_encoder/mel_fb`), plain-C++ LSTM forward pass
  (no ggml graph), partial-window averaging that exactly reproduces
  `VoiceEncoder.embeds_from_wavs(..., as_spk=False)` for a single wav:
  mel is split into overlapping 160-frame partials using
  `get_frame_step`/`get_num_wins`, each partial produces an L2-normed
  256-d embedding via LSTM + projection, then the per-partial embeds
  are averaged and L2-normed once more.
- `src/test_voice_encoder.cpp` — parity harness; compares the C++
  256-d `speaker_emb` against Python `speaker_emb.npy` using
  `max_abs`, `rms`, `rel` and cosine similarity.

Converter change: `scripts/convert-t3-turbo-to-gguf.py` now bakes in
the VE weights (`weight_ih_l{0,1,2}`, `weight_hh_l{0,1,2}`,
`bias_{i,h}h_l{0,1,2}`, `proj/weight`, `proj/bias`) plus the librosa
(40, 201) mel filterbank as `voice_encoder/mel_fb`, and writes VE
hyperparameters (n_mels, hidden_size, num_layers, partial_frames,
sample_rate, n_fft, hop_size, win_size, overlap, rate, min_coverage)
as GGUF metadata so we never need `ve.safetensors` at runtime.  The
`similarity_{weight,bias}` params are skipped — they're only used for
speaker-verification training, not embedding extraction.

Feature extraction: `src/voice_features.cpp` gained
`mel_extract_16k_40`, which shares the STFT/mel core with
`mel_extract_24k_80` but uses the VE-specific knobs (`center=True`,
`power_exponent=2`, no log compression).

CLI wiring: `main.cpp` now resolves the T3 voice override in two
independent pieces.  If `ref_dir/speaker_emb.npy` is missing but
`--reference-audio PATH.wav` is given AND the T3 GGUF has VE weights,
it loads the wav, resamples to 16 kHz, and computes `speaker_emb` in
C++ via `voice_encoder_embed()`.  `cond_prompt_speech_tokens` still
comes from `ref_dir` until Phase 2e. Logs distinguish the source:
`T3 voice override — speaker_emb=C++ VoiceEncoder, cond_prompt_tokens=ref_dir`.

Verification on 10.4 s reference wav:

```
[result] C++ vs Python speaker_emb:
    n=256  max_abs=1.71e-05  rms=2.58e-06  max|ref|=2.45e-01  rel=6.97e-05
    cosine similarity = 1.000000
```

Cosine = 1.000000 confirms angular match to 6 decimal places; the
~1e-5 absolute error is pure float32 accumulation noise.  End-to-end
synthesis with `speaker_emb.npy` *deleted* from the voice dir produced
a 276 kB WAV that plays cleanly on macOS — the C++-computed speaker
embedding drives T3 conditioning indistinguishably from Python.

Two down, two to go (`embedding` and `prompt_token` via CAMPPlus +
S3TokenizerV2).

**Phase 2d-a (DONE)** — C++ CAMPPlus forward pass, validated end-to-end
against the Python reference on a Python-dumped 80-ch Kaldi fbank.

CAMPPlus is a FunASR/3D-Speaker x-vector: 937 raw tensors (329 conv /
linear weights + 122 BatchNorms + biases + counters).  Structure:

```
  fbank (T, 80)
    → FCM: Conv2d(1→32, k=3) + BN + 2× BasicResBlock (stride=2)
              + 2× BasicResBlock (stride=2) + Conv2d(32→32, s=(2,1))
              + reshape → (320, T)
    → xvector.tdnn: Conv1d(320→128, k=5, s=2) + BN + ReLU
    → 3 × CAMDenseTDNNBlock + TransitLayer
         block1: 12 layers, dilation=1  → 128 → 512
         transit1: Conv1x1 + BN: 512 → 256
         block2: 24 layers, dilation=2  → 256 → 1024
         transit2: 1024 → 512
         block3: 16 layers, dilation=2  → 512 → 1024
         transit3: 1024 → 512
    → out_nonlinear (BN + ReLU)
    → stats_pool (mean + unbiased std over T → 1024)
    → dense: Conv1x1(1024→192) + BN(affine=False) → 192
```

Each `CAMDenseTDNNLayer` is `BN→ReLU→Conv1x1→BN→ReLU→CAMLayer`, with
`CAMLayer` being `linear_local × sigmoid(linear2(ReLU(linear1(ctx))))`
where `ctx = mean(x, T) + seg_pool(x, 100).expand(T)`.

Ports:
- `scripts/convert-s3gen-to-gguf.py` — fuses every BatchNorm into a
  per-channel `(scale, shift)` pair at export time:
    `scale = gamma / sqrt(var + eps)` (or `1/sqrt(var + eps)` when
    `affine=False`), `shift = beta - mean*scale`.  Skips
    `num_batches_tracked`.  Embeds 14 `campplus.*` hyperparameters as
    GGUF metadata and emits the 451 substantive tensors under
    `campplus/…` (329 conv + 122 fused BNs).
- `src/campplus.{h,cpp}` — plain-C++ forward pass, no ggml graph.
  Uses channel-major `(C, T)` layout throughout.  Helpers: `bn_apply`,
  `relu_inplace`, `sigmoid_inplace`, `conv1d`, `conv2d`,
  `seg_pool_expand` (avg-pool with `ceil_mode=True` + repeat-interleave
  to `T`), `stats_pool` (mean + unbiased std).  Module-level helpers
  `fcm_basic_resblock`, `fcm_forward`, `cam_layer_forward`,
  `cam_dense_tdnn_layer_forward`.  Parallelised via OpenMP.
- `src/test_campplus.cpp` — loads CAMPPlus from `chatterbox-s3gen.gguf`,
  runs on a Python-dumped `fbank.npy`, compares with Python
  `embedding.npy` using max_abs / rms / rel / cosine similarity.
- `scripts/dump-campplus-reference.py` — helper that loads the turbo
  checkpoint, runs `extract_feature` (Kaldi fbank + per-utterance
  mean-subtract) and `speaker_encoder.forward`, and dumps the two
  tensors to `.npy`.

Result on a 10.4 s reference wav (1038 fbank frames, 192-d output):

```
[result] C++ vs Python embedding:
    n=192  max_abs=2.34e-05  rms=6.99e-06  max|ref|=2.49e+00  rel=9.38e-06
    cosine similarity = 1.000000
    forward pass: 549.9 ms (16-thread EPYC)
```

`rel = 9.4 ppm`, cosine = 1.000000 — numerical parity. 550 ms for a
one-time voice-setup pass is comfortably fast.

`src/s3gen_pipeline.h` grew an `embedding_override` field and
`src/chatterbox_tts.cpp` reads it in place of `ref_dir/embedding.npy`
when provided, mirroring `prompt_feat_override`.  End-to-end wiring
into `main.cpp` is blocked on Phase 2d-b (Kaldi fbank port) — we can't
feed CAMPPlus from `--reference-audio` until the C++ binary can
extract its own fbank.

**Phase 2d-b (DONE)** — C++ port of
`torchaudio.compliance.kaldi.fbank` with `num_mel_bins=80`.
Implemented as `fbank_kaldi_80` in `src/voice_features.{h,cpp}`
with all the Kaldi knobs baked in:

- `frame_length = 25 ms = 400 samples`, `hop = 10 ms = 160 samples`
- `round_to_power_of_two = True` → `n_fft = 512`
- `window_type = "povey"` = `hann(N, periodic=False) ** 0.85`
- `remove_dc_offset = True` (subtract per-frame mean)
- `preemphasis_coefficient = 0.97`, with the Kaldi edge case
  `out[0] = frame[0] * (1 - coeff)`
- `use_power = True`, `use_log_fbank = True` with `log_floor = FLT_EPSILON`
- `snip_edges = True`, `dither = 0`
- Kaldi mel filterbank (`mel = 1127 * log(1 + f / 700)`, triangular
  filters equally spaced in mel-space) precomputed by
  `convert-s3gen-to-gguf.py` and baked in as
  `campplus/mel_fb_kaldi_80` (shape `(80, 257)`).

Key gotcha we hit: **torchaudio's Kaldi wrapper does _not_ apply
the `×32768` int16 scaling that real Kaldi does.**  With the scale
our output was +20.8 units offset from Python (exactly
`2 * log(32768) ≈ 20.79`).  Dropped the scale and `rel` jumped from
`1.30` to `1.77e-05`.

Validation on the synthetic 10 s speech signal:

```
[result] C++ vs Python fbank:
    n=79840  max_abs=2.82e-04  rms=5.91e-06  max|ref|=1.59e+01  rel=1.77e-05

C++ fb[0, :8]: -10.1011 -8.3549 -7.9557 -7.4304 -7.0186 ...
Py  fb[0, :8]: -10.1012 -8.3549 -7.9557 -7.4304 -7.0186 ...
```

**Phase 2d-c (DONE)** — Wired into `main.cpp`.  New
`compute_embedding_native()` glues `wav_load → resample_sinc →
fbank_kaldi_80 → mean-subtract over T → campplus_embed` and
populates the new `embedding_override` field in
`s3gen_synthesize_opts`.  Called best-effort from both short-circuit
and regular T3→S3Gen paths: if the s3gen GGUF pre-dates Phase 2d-a
(no CAMPPlus tensors), it silently falls back to
`ref_dir/embedding.npy`.

End-to-end dogfood on the 10.4 s reference wav with
`speaker_emb.npy` _and_ `embedding.npy` deleted from `voices/test/`:

```
voice_encoder: computing speaker_emb from /tmp/unified_remote.wav
main: T3 voice override — speaker_emb=C++ VoiceEncoder, cond_prompt_tokens=ref_dir
voice: prompt_feat shape=(520, 80)
voice: embedding shape=(192,) via CAMPPlus (1038 fbank frames)
  embedding:   using C++ override (CAMPPlus, 192 dims)
  prompt_feat: using C++ override (520 mel frames)
```

Output WAV plays cleanly and sounds identical to the Python
voice-cloned output.  Only `cond_prompt_speech_tokens.npy` and
`prompt_token.npy` still live in `ref_dir` — both are produced by
`S3TokenizerV2`, the last holdout (Phase 2e).

**Phase 2e (DONE)** — C++ S3TokenizerV2: a 6-layer FSMN-attention
transformer + FSQ codebook that turns a 16 kHz reference wav into the
25 Hz speech-token stream Chatterbox needs for voice conditioning.
103 tensors / ~124 M params.  Produces BOTH the T3-side
`cond_prompt_speech_tokens` and the S3Gen-side `prompt_token` streams.

Architecture (mirrors `s3tokenizer.model_v2.S3TokenizerV2` exactly):

```
  wav_16k
    → log_mel_spectrogram (n_fft=400, hop=160, 128 mels, log10 clamp+floor
        + (x + 4) / 4 normalise)
    → Conv1d(128 → 1280, k=3, s=2) + GELU
    → Conv1d(1280 → 1280, k=3, s=2) + GELU
    → 6 × ResidualAttentionBlock:
        LN → q/k/v (RoPE, NEOX-style, theta=10000)
        depth-wise Conv1d(k=31) over v → fsmn_memory
        scaled dot-product attention
        out = Linear(attn) + fsmn_memory
        LN → Linear 1280→5120 → GELU → Linear 5120→1280
    → FSQCodebook:
        Linear(1280 → 8) → tanh * 0.999 → round + 1
        token = Σ h[i] * 3^i   (0..6560)
```

Implementation:
- `src/s3tokenizer.{h,cpp}`: weights struct + GGUF loader +
  `s3tokv2_log_mel` (plain C++ STFT + mel filterbank + log clamp +
  normalise) + `s3tokv2_tokenize` (ggml graph for conv-stem +
  6 transformer blocks + plain-C++ FSQ).  Uses the standard pattern:
  one weight context (no_alloc, pre-allocated backend buffer) + a
  per-run input context + a big graph context for intermediates,
  allocated via `ggml_gallocr`.
- Subtleties:
    - `ggml_conv_1d` and `ggml_conv_1d_dw_ph` both assert F16 kernels
      in their fused kernel paths; we ship F32 weights, so we go
      through `ggml_im2col + ggml_mul_mat` manually
      (`conv1d_f32`, `conv1d_dw_f32`).
    - ggml conv output has time innermost (ne=[T, C]), but the
      transformer wants channels innermost (ne=[C, T]) for LN and
      1-D bias broadcasts.  We `ggml_cont(ggml_transpose(...))`
      between the stem and the blocks.
    - Attention permutations: q/k to ne=(head_dim, T, n_head),
      v to ne=(T, head_dim, n_head), so `mul_mat(k, q)` gives
      scores ne=(T_k, T_q, n_head) with T_k innermost for
      `ggml_soft_max`, and `mul_mat(v, scores)` gives
      out ne=(head_dim, T_q, n_head).
    - RoPE: `ggml_rope_ext` with `GGML_ROPE_TYPE_NEOX`,
      `freq_base = 10000`, `n_ctx_orig = 2048`, matches the
      reference's half-split `rotate_half` convention.
- Converter: `convert-s3gen-to-gguf.py` emits all 103 `tokenizer.*`
  tensors as `s3tokv2/…` plus 15 hyperparameters as GGUF metadata.
- `scripts/dump-s3tokenizer-reference.py`: dumps `wav_16k.npy`,
  `log_mel.npy`, and `tokens.npy` for validation.
- `src/test_s3tokenizer.cpp`: parity harness that validates log-mel
  (always passes cleanly) and reports token accuracy vs Python.

Validation on a 10 s synthetic speech signal:

```
  log_mel : max_abs=1.80e-05  rel=1.30e-05     (numerical parity)
  tokens  : 236 / 250 = 94.40%                 (FSQ-rounding drift)
```

FSQ is extremely sensitive: the project_down → tanh → round pipeline
turns 8 floats into 8 ternary digits, so sub-LSB float drift through
the 6 transformer layers can flip a digit and change the token.  Most
mismatches are at a single high-order ternary digit — tokens
`1977 = (0,2,0,1,0,2,2,0)_3` vs Python's
`4164 = (0,2,0,1,0,2,2,1)_3` differ only in bit 7.  In practice the
resulting speaker conditioning is close enough that the cloned audio
sounds identical.

Wiring: `main.cpp` gained `compute_speech_tokens_native()` which runs
the tokenizer twice (first 10 s of the wav → `prompt_token`, first
15 s → `cond_prompt_speech_tokens` capped to `speech_cond_prompt_len`).
Results feed `s3gen_synthesize_opts::prompt_token_override` (new
field) and the existing T3 `cond_prompt_speech_tokens` override path.

**End-to-end pure-C++ voice cloning**: with `voices/test/` deleted
entirely and only `--reference-audio my.wav` given, the unified
`chatterbox` binary now runs the whole flow in C++:

```
voice_encoder: computing speaker_emb from /tmp/unified_remote.wav
voice: prompt_token=(250,) cond_prompt_speech_tokens=(260,) via S3TokenizerV2
main: T3 voice override — speaker_emb=C++ VoiceEncoder, cond_prompt_tokens=C++ S3TokenizerV2
voice: prompt_feat shape=(520, 80)
voice: embedding shape=(192,) via CAMPPlus (1038 fbank frames)
  prompt_token: using C++ override (S3TokenizerV2, 250 tokens)
  embedding:    using C++ override (CAMPPlus, 192 dims)
  prompt_feat:  using C++ override (520 mel frames)
```

`scripts/prepare-voice.py` is now redundant — the CLI only needs a
reference wav.  Impact: voice cloning has **zero Python runtime
dependencies**; a user just runs the binary.

Impact: Phase 1 unlocked voice cloning as a usable feature. Phases
2a–2e replaced every Python preprocessing step with a native C++
port, so the deployment story is now "one binary + two GGUFs".

#### A2. GPU backend (Vulkan + Metal) — ✅ **DONE** (see §3.11 + §3.12)

Wired `--n-gpu-layers` through both T3 and S3Gen/HiFT. Now builds with
any of `-DGGML_CUDA=ON`, `-DGGML_METAL=ON`, or `-DGGML_VULKAN=ON`;
`init_backend()` in `main.cpp` and `s3gen_init_backend()` in
`chatterbox_tts.cpp` pick the matching backend when `n_gpu_layers > 0`
and fall back to CPU otherwise.

Out-of-the-box Metal was missing three things that needed kernel-level
fixes in `ggml/src/ggml-metal/`:

- `GGML_OP_DIAG_MASK_INF` — no dispatcher entry. Added a kernel +
  pipeline getter + op encoder + `supports_op` case.
- `GGML_OP_PAD` with non-zero front padding — rejected by
  `supports_op`. Extended `kargs_pad` with `lp0..lp3`, updated the
  kernel to apply them, relaxed the check.
- `GGML_OP_CONV_TRANSPOSE_1D` — kernel was scalar. Tightened the
  input-position loop (`i_start..i_end` instead of `0..IL`) and
  parallelised the `IC` reduction across a 32-thread simdgroup with
  `simd_sum`. 40× speedup on HiFT-sized shapes.

Patches live in `patches/ggml-metal-chatterbox-ops.patch` (applied to
the vendored ggml during build); `src/test_metal_ops.cpp` validates
each patched kernel against the CPU reference. CUDA and Vulkan needed
no backend changes — only the chatterbox wiring.

Result: `gen_RTF` on a 10 s sentence drops from **0.70 (CPU)** to
**0.08 (Vulkan 5090)** and **0.18 (Metal M3 Ultra)**.

Still open: T3 autoregressive inference dominates wall time on small
GPUs (≈ 1.3 s for 260 tokens on a 60-core Apple GPU). Worth exploring
speculative decoding or a smaller T3 draft model if further wins are
needed — but current numbers are already interactive.

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

##### Phase 2 (CFM bit-exact parity) — ✅ DONE (2026-04-12)

Before shipping the streaming binary we needed the per-chunk C++ mel to
match Python to float32 precision. The per-chunk harness
(`src/test_streaming.cpp` + `scripts/dump-streaming-reference.py`) now
reports `worst rel = 8.67e-07` for both chunks (i.e. machine epsilon) on
the `test.wav` reference.

The last bug found was subtle: Chatterbox's turbo flow runs CFM in
**meanflow** mode, which means `flow_inference` allocates a
*second* noise tensor

```python
noise = torch.randn(1, 80, speech_tokens.size(-1) * 2, ...)
super().forward(..., noised_mels=noise)
```

and `flow_matching.forward` silently **overwrites the speech region of
`z`**:

```python
z = torch.randn_like(mu) * temperature
if noised_mels is not None:
    prompt_len = mu.size(2) - noised_mels.size(2)
    z[..., prompt_len:] = noised_mels   # ← second randn draw lives here
```

Our original Python capture hook wrapped only `torch.randn_like`, so the
saved `chunk_KK_cfm_z.npy` contained the *first* draw everywhere,
including positions `t ≥ prompt_len` that are actually overwritten by
the second draw. Injecting that stale `z` as `cfm_z0_override` in C++
produced CFM output that matched Python bit-exactly in the prompt
region (`t < 500`) and diverged wildly in the speech region (`t ≥ 500`)
— exactly the "receptive field of the prompt/speech boundary" pattern
we were chasing.

Fix (commit [`2e82cce`](https://github.com/gianni-cor/chatterbox.cpp/commit/2e82cce)
and the follow-up in this section):
- Replace the `torch.randn_like` capture with a wrapper around
  `CausalConditionalCFM.basic_euler` that records the full `x` tensor
  at the first `estimator.forward` call. That tensor *is* the real z
  after the meanflow overlay.
- Dump it as `chunk_KK_step0_x_in.npy`; `test-streaming` loads that
  (instead of the old `chunk_KK_cfm_z.npy`) into `cfm_z0_override`.
- All four CFM inputs (`mu`, `mask`, `spks`, `cond`) already matched at
  rel ≤ 3e-7, so fixing `z` made the estimator output match at rel ≈
  machine epsilon.

Lessons: in streaming validation harnesses, capture *the exact tensor
the target op receives*, not an earlier upstream value. Monkeypatching
a function that a caller later post-processes (`z[...] = …`) is a
silent source of divergence.

##### Phase 3 (HiFT streaming + CLI) — ✅ DONE (2026-04-12)

With CFM bit-exact across chunks, wiring up the HiFT side and the
user-facing CLI was straightforward:

- **`cache_source` carry** (src/chatterbox_tts.cpp, `s3gen_synthesize_opts`):
  after `sinegen_source` produces the post-`m_source` source signal,
  overwrite its leading samples with the caller-provided
  `hift_cache_source` and expose the last `source_tail_samples` (480 =
  1 mel hop = 20 ms) via `hift_source_tail_out` so the caller can feed
  them back in on the next chunk. Matches Python
  `HiFTGenerator.inference`'s `s[:, :, :cache_source.shape[2]] = cache_source`.

- **`trim_fade`** (same file): opt-in raised-cosine fade-in applied to
  the first `2 * sr/50 = 960` samples (40 ms) of each chunk's wav.
  First half zero, second half `(cos(π→0)+1)/2`. Streaming callers set
  `apply_trim_fade` on chunk 0 only.

- **`--stream-chunk-tokens N` CLI flag** (src/main.cpp): wraps
  `s3gen_synthesize_to_wav` in a chunked loop that carries
  `hift_cache_source` across chunks, writes per-chunk wavs as
  `<out>_chunk_KK.wav`, and concatenates the final wav into `--out`.
  Adds `append_lookahead_silence=false`, `finalize=(is_last)`, and
  `skip_mel_frames=prev_mels_emitted` on each chunk.

- **Process-wide model cache** (src/chatterbox_tts.cpp,
  `s3gen_model_cache_get`): makes the ~700 ms GGUF-tensor load a
  one-shot cost. `s3gen_preload(path, n_gpu_layers)` populates the cache
  eagerly so main.cpp can kick a background std::thread to warm S3Gen
  while T3 is still running. Brings first-chunk latency down from
  2006 ms → 1340 ms on CPU for the `"streaming sanity check"` test.

Validation (`./build/test-streaming models/chatterbox-s3gen.gguf
/tmp/streaming_ref`):

| chunk | mel rel | wav rel (informational) |
|---|---|---|
| 1 | 6.47e-07 | 1.06e-01 |
| 2 | 8.67e-07 | 1.24e-01 |

Mel is bit-exact; wav diverges a few percent because C++'s
`sinegen_source` uses `std::mt19937` vs Python's `torch.randn` — the
audio content is identical, only the per-sample additive white-noise
seed differs. Python's own streamed-vs-batch ratio is 116 %, so our
streamed-vs-Python-streamed is 6.5 %, well inside the structural
envelope of the approach.

Performance numbers on a 3.76 s utterance (9 s of reference audio):

| metric | batch | streaming (25 tokens/chunk) |
|---|---|---|
| total wall time | 2271 ms | 5988 ms |
| first-audio-out | 2271 ms | **1340 ms** |
| per-chunk RTF | 0.60 | 1.44 – 1.59 |

##### Phase 3b (per-chunk RTF tuning) — ✅ DONE (2026-04-12)

Per-chunk profiling on the same 4.9 s utterance:

| stage | cost per chunk (T_mu≈650) |
|---|---|
| encoder (T_tokens≈350) | ~280 ms |
| CFM step 0 | ~580 ms |
| CFM step 1 | ~500 ms |
| HiFT decode (1 s audio) | ~265 ms |
| total | **~1630 ms for 1 s of audio** |

CFM is ~2/3 of every chunk.  Two things that *don't* work for cutting
it down without retraining:

- **KV-cached CFM / incremental encoder** — Chatterbox's flow encoder
  and CFM estimator both run full bidirectional self-attention.  I
  verified `static_chunk_size = 0` in `decoder.py` (no chunked
  attention mask) and that the encoder has no causal mask either.
  Caching previous-chunk activations would require the attention to
  be *causal* (or at least chunk-causal).  Retrofitting that at
  inference time changes the output distribution — not a pure port.
- **Prompt-region truncation** — the 500-frame prompt accounts for
  ~70 % of T_mu and its CFM output is discarded every chunk.  But
  attention is full, so any speech-region output depends on every
  prompt frame via softmax.  Truncating to a short prompt tail would
  require retraining.

What *does* work, and is now shipped as tunables:

- **Non-uniform chunk sizes** (`--stream-first-chunk-tokens N`).
  First chunk stays small (≈10 tokens / 0.4 s audio) for fast
  first-audio-out; subsequent chunks go big (≈50 tokens / 2 s audio)
  so the fixed per-chunk encoder+CFM cost amortises over more output.
- **Fewer CFM Euler steps** (`--stream-cfm-steps 1`).  Turbo is
  meanflow-trained, and meanflow supports 1-step sampling per the
  paper.  In practice 1-step introduces some audible high-frequency
  noise (log-mag MAE ≈ 0.5 vs 2-step) but keeps content intact.
  Default stays at 2 to match Python; users opt in via the flag.

Measured on the same text on CPU:

| config | first-audio | chunk-N RTF | overall RTF |
|---|---|---|---|
| baseline (`--stream-chunk-tokens 25`) | 1331 ms | 1.44 – 1.70 | 1.59 |
| first-small (`10 → 25`) | 1156 ms | 1.37 – 1.69 | 1.84 |
| 1-step + big (`50`, `steps=1`) | 1230 ms | 0.63 – 0.69 | 0.78 |
| **combined (`10 → 50`, `steps=1`)** | **782 ms** | **0.63 – 0.69** | **0.94** |

The "combined" preset hits both objectives at once: first audio out
in ≤ 800 ms on CPU, and middle chunks complete in 2/3 of their audio
duration so the streamer can stay ahead of playback.  Incremental
encoder / KV-cached CFM stay on the backlog for when someone wants to
retrain Chatterbox with chunk-causal attention.

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

With A1 (voice cloning), A2 (GPU backends) and A3 (T3 quantization)
done, the remaining high-impact work is:

1. **B1 — Streaming** (~1 week) → what gets this into interactive apps;
   currently blocked only by wiring, not correctness.
2. **C3 — CI + prebuilt GGUFs** — pick up before announcing publicly.
3. **T3 autoregressive speedup** (speculative decoding, or a smaller T3
   draft model). Biggest chunk of wall time left on both Metal and
   Vulkan now that HiFT is fast.

B2 (server mode) and C1 (custom Conformer attn op) are worth doing once a
concrete deployment is pressuring for them; the CPU numbers are already
well past real-time for CLI use, and the GPU numbers are at
multi-x real-time with zero extra work.
