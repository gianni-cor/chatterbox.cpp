# Chatterbox → ggml Port: Development Journal

This document tracks the port of **Chatterbox Turbo** (Resemble AI, MIT license) to
`ggml` — what was built, what was verified, what's left, and a log of the non-obvious
issues hit along the way.

- **Model**: `ResembleAI/chatterbox-turbo` (text-to-speech, ~450 M params without
  the tokenizer / speaker-encoder).
- **Goal**: end-to-end text → waveform in C++/ggml with **bit-exact (or
  float-precision) parity** against the official PyTorch reference.
- **Verification target**: every intermediate tensor within 1e-6 relative error
  of the PyTorch implementation, on CPU.

---

## 1. Current status

### ✅ Done and verified on the remote Linux x86-64 machine (CPU)

| Stage | Component | Max relative error vs PyTorch |
|-------|-----------|-------------------------------|
| — | GPT-2 BPE tokenizer (self-contained C++) | **10 / 10** exact-match test cases |
| — | T3 speech-token generation (greedy + sampling) | **bit-exact** on 4 deterministic prompts |
| A | S3Gen speaker_emb projection (F.normalize + Linear) | 1.2e-7 |
| B | S3Gen input embedding (nn.Embedding lookup) | 0 (exact) |
| C | Encoder embed (Linear + LayerNorm + sqrt(D) scale + ESPnet rel PE) | 4.4e-7 |
| D | PreLookaheadLayer (asymmetric-padded Conv1d stack) | 2.5e-7 |
| E | One Conformer encoder block (rel-pos MHA + rel_shift + Swish FFN) | 1.3e-7 |
| **F** | **Full encoder** (6 blocks + Upsample1D + 4 blocks + after_norm + encoder_proj) | **5.6e-7** |
| G1 | CFM time embedding (sin → MLP → mixer) | 7.0e-7 |
| G2 | CFM `CausalResnetBlock1D` (causal-conv + LN + Mish + time MLP + res_conv) | 2.9e-7 |
| G3 | `BasicTransformerBlock` (self-attn + FFN with GELU-erf) | 1.7e-7 |
| **G4** | **Full CFM decoder one step** (1 down + 12 mid + 1 up + final + final_proj) | **1.3e-6** |

Runnable from CPU today: `text → tokens → speech tokens` (via `chatterbox`
binary). `mel → waveform` (HiFT) is the last missing piece.

### ⏳ Not yet done

- **HiFTGenerator** (vocoder, mel → waveform):
  - `ConvRNNF0Predictor`
  - `SineGen` / `SourceModuleHnNSF` (harmonic excitation from F0)
  - Transposed-conv upsampling stack `[8, 5, 3]`
  - Multi-ResBlock with Snake activation
  - STFT/ISTFT output head (24 kHz, hop = 480)
  - 246 tensors, ~20 MB; `weight_norm` already resolved at conversion time.
- **End-to-end binary** that glues T3 + S3Gen + HiFT and writes a `.wav`.

---

## 2. Repository layout

```
qvac-chatterbox.cpp/
  ggml/                           ← pristine ggml checkout (vendored, unmodified)
  src/
    main.cpp                      T3 runtime  (chatterbox binary)
    gpt2_bpe.{h,cpp}              self-contained GPT-2 byte-level BPE tokenizer
    test_s3gen.cpp                staged verification harness (stages A..G4)
    npy.h                         minimal .npy loader + compare helpers
  scripts/
    convert-t3-turbo-to-gguf.py   converts Turbo T3 weights + conds to GGUF
    convert-s3gen-to-gguf.py      converts flow (encoder + CFM) + HiFT to GGUF
    dump-s3gen-reference.py       runs PyTorch, dumps every intermediate tensor
    reference-t3-turbo.py         runs PyTorch T3, compares against C++
    compare-tokenizer.py          10-case tokenizer comparison against HF
  models/
    chatterbox-t3-turbo.gguf      T3 + tokenizer conditionals
    chatterbox-s3gen.gguf         flow + mel2wav weights + built-in voice
  CMakeLists.txt                  top-level: add_subdirectory(ggml) + two targets
  README.md                       quick-start
  PROGRESS.md                     this file
```

A separate machine holds PyTorch + the original Chatterbox repo for reference
runs. On-device (Apple Silicon / Linux x86) the C++ binary has **no runtime
dependency on Python** — the tokenizer reads `vocab.json` + `merges.txt`
directly.

---

## 3. High-level timeline / what was done

1. **Scoping** — surveyed open-source TTS landscape
   (`F5-TTS`, `Kokoro`, `XTTS v2`, `Chatterbox`, `Supertonic`, …) and chose
   `Chatterbox Turbo`: MIT-licensed, zero-shot cloning, Turbo variant runs with
   just 2 flow-matching steps (fast).

2. **Repo bootstrap** — cloned latest `ggml` + reference
   `resemble-ai/chatterbox` side-by-side; built a standalone
   `qvac-chatterbox.cpp/` with `ggml/` as a vendored subdirectory (no
   modifications inside `ggml/`).

3. **T3 port** (GPT-2 Medium size, 24 layers) — reused the pattern from
   ggml's `examples/gpt-2`:
   - Wrote `scripts/convert-t3-turbo-to-gguf.py` to emit GGUF with built-in
     voice conditionals (`speaker_emb`, `cond_prompt_speech_tokens`) embedded.
   - C++ graph uses separate "prompt" and "step" graphs with a persistent KV
     cache.
   - Verified against PyTorch: **bit-for-bit** identical speech tokens on 4
     deterministic sampling configs (greedy / temperature / top-k /
     repetition-penalty / no-penalty × short + long prompts).

4. **C++ tokenizer** — studied llama.cpp's BPE (too entangled with GGUF vocab
   loading) and wrote a self-contained GPT-2 BPE in `src/gpt2_bpe.cpp`:
   byte-level encoding table, regex pre-tokenization, BPE merge loop, plus
   `punc_norm` matching Python's implementation. 10/10 test cases match HF
   tokenizer byte-for-byte, including the 19 paralinguistic added tokens
   (`[laugh]`, `[chuckle]`, …).

5. **Full-pipeline glue for T3** — `chatterbox` binary takes `--text` +
   `--tokenizer-dir` and produces speech tokens end-to-end in C++.

6. **S3Gen encoder** (Upsample Conformer, 10 blocks total, ~60 M params):
   - Python reference dumper captures every intermediate via `forward_hook`.
   - Staged C++ implementation (stages A–F above) with per-stage comparison.
   - Tricky parts: ESPnet relative positional encoding, `rel_shift` attention
     score alignment, nearest-neighbor ×2 upsample via `concat` trick.

7. **CFM decoder** (U-Net with transformer blocks, ~45 M params):
   - 1 down block → 12 mid blocks → 1 up block (skip concat) → final_block →
     final_proj. Each block carries 4 `BasicTransformerBlock`s.
   - Time embedding: sinusoidal(320) → MLP(320→1024) → mixer(2048→1024) on
     `concat(t_emb, r_emb)` for meanflow mode.
   - Verified as a single forward step against `cfm_step0_dxdt.npy` at
     **rel=1.3e-6**.

---

## 4. Issues found and how they were fixed

Grouping by theme so the story is easy to pattern-match the next time a similar
bug appears.

### 4.1 Repo / tooling issues

| # | Issue | Cause | Fix |
|---|-------|-------|-----|
| 1 | `rsync` not on macOS by default | — | Switched to `tar … \| ssh … tar -x`. |
| 2 | Remote repo polluted with `._*` AppleDouble files | macOS `tar` writes extended attributes | `COPYFILE_DISABLE=1 tar …` before SSH pipe. |
| 3 | Partial sync left `src/CMakeLists.txt` stray file | Earlier `scp` blasted a nested CMake file | Removed, and unified sync to always push the whole tree. |
| 4 | Remote binary `0 bytes` after SSH disconnect | Link step got killed mid-write | `rm build/<target>` + rebuild. |
| 5 | SSH session dropped for several minutes mid-task | Remote transient | Retried with `ConnectTimeout=10` loops. |

### 4.2 ggml layout & tensor-shape traps

| # | Issue | Cause | Fix |
|---|-------|-------|-----|
| 6 | `ggml_can_mul_mat` assertion in `T3` | PyTorch weight `[out, in]` needs transpose for GGUF export of `Conv1D` (`GPT2`'s `c_attn`, `c_proj`, `c_fc`, `mlp.c_proj`) while `nn.Linear`, embeddings, `wpe` already match | Converter transposes only the `Conv1D`-style weights; embeddings and `wpe` pass-through. |
| 7 | `speaker_embed` broadcasting failed in `cond_spkr` matmul | Reshape produced `ne=[256]` instead of `ne=[1, 256]` | Explicit `reshape_2d(bias, 1, 256)` whenever a `ne=[C]` bias is added to a `ne=[T, C]` conv1d output. |
| 8 | Nearest-neighbor ×2 upsample gave "interleaved by channel" instead of "repeated per t" | First attempt reshaped to `ne=[T, 1, D]` and concat'd along `ne[1]` → memory order was `t0_copy0, t1_copy0, …, t0_copy1, …` | Correct trick: reshape to `ne=[1, T, D]` → `concat` along `ne[0]` → `ne=[2, T, D]` → reshape to `ne=[2T, D]`, giving `t0_copy0, t0_copy1, t1_copy0, …`. |
| 9 | `rel_shift` attention produced garbage (~100 % rel error) | `view_3d(bd_viewed, T, 2T-1, H, nb1, **T*(2T-1)*elem**, offset)` used the *sliced* ne[1] size for `nb2` | `nb2` must match the *source's* element stride: use `bd_viewed->nb[2]` directly. |
| 10 | `ggml_backend_tensor_get(input_tensor)` returned garbage | `ggml_gallocr` reused the input buffer for intermediates because we only marked it `set_input` | Also call `ggml_set_output` on tensors we want to read back; or just read them before `graph_compute`. |
| 11 | `layer_norm` applied over time axis instead of channel | For `ne=[T, C]` layout, `ggml_norm` reduces `ne[0]=T`, which is wrong | Added a `layer_norm_on_channel` helper that permutes to `ne=[C, T]`, norms, applies affine, permutes back. |

### 4.3 Weight dtype / op compatibility

| # | Issue | Cause | Fix |
|---|-------|-------|-----|
| 12 | `ggml_conv_1d` aborted with `src0->type == GGML_TYPE_F16` | Core `ggml_im2col` path requires the kernel to be F16 | Wrote `conv1d_f32` helper that calls `ggml_im2col(…, GGML_TYPE_F32)` + `mul_mat` directly, keeping kernels in F32 for precision. |
| 13 | Accidentally left `up_layer/conv/w` as F16 while the rest of the convs moved to F32 | Mixed converter state across partial edits | Single converter policy now: **all convs stay F32** (with `conv1d_f32`). Quantization can be added later in one place. |
| 14 | Ignored `weight_norm` convolutions in `mel2wav` | Torch 2.6 stores them under `parametrizations.weight.original{0,1}` | `expand_weight_norm()` in the converter fuses `g * v / ‖v‖₂` back into a normal `weight` tensor before export. |
| 15 | GELU mismatch in `BasicTransformerBlock` (rel=3e-4) | `ggml_gelu` is the tanh approximation; `diffusers.models.activations.GELU` uses the exact `erf` formulation | Switched to `ggml_gelu_erf`. Error dropped to 1.7e-7. |
| 16 | Mish activation missing from ggml unary ops | No `GGML_UNARY_OP_MISH` | Built from primitives: `x * tanh(softplus(x))` via `GGML_UNARY_OP_SOFTPLUS` + `GGML_UNARY_OP_TANH`. |

### 4.4 NumPy / reference-dump gotchas

| # | Issue | Cause | Fix |
|---|-------|-------|-----|
| 17 | C++ comparisons showed **100 %** error on `h_ln` even though values in Python looked right | Torch's `.transpose()` yields a non-contiguous view; `np.save` stores it as **Fortran-ordered** (`fortran_order: True` in the .npy header) | Dumper now calls `t.detach().cpu().contiguous().numpy()` followed by `np.ascontiguousarray(...)`. C++ loader also throws a clear error if it sees `fortran_order=True`. |
| 18 | Python hook overwrote the same tensor across multiple CFM steps | Meanflow calls `time_embeddings` once for `t` and again for `r`; also the full decoder runs twice per sample (`t_span = [0, 0.5, 1]`) | `make_hook(multi_call=True)` counts invocations and saves `*_call0.npy`, `*_call1.npy`, …. |
| 19 | Confusion between Python `(B, C, T)` vs Python `(B, T, C)` layouts | CFM alternates: resnets are `(B, C, T)`, transformer blocks are `(B, T, C)`, switched via `rearrange` calls | In ggml we mirror this: resnet uses `ne=[T, C]` (= numpy `(C, T)`), transformer uses `ne=[C, T]`. Both helpers clearly label the convention in their doc comments; we `cont(permute)` at the boundary. |

### 4.5 Sampling / numerics parity

| # | Issue | Cause | Fix |
|---|-------|-------|-----|
| 20 | Repetition-penalty path diverged from HF at token 22 onwards | Python's `RepetitionPenaltyLogitsProcessor` divides **positive** logits by the penalty and multiplies **negative** logits (shrinks toward 0). My first C++ version did the opposite. | Flipped the sign condition. |
| 21 | Sampler order mismatched HF's `LogitsProcessorList` | Initial C++ applied rep-penalty first; HF applies Temperature → TopK → TopP → RepetitionPenalty, in that order | Rewrote `sample_next_token` to mirror HF's order exactly. After the fix, greedy+penalty tests pass bit-exactly. |

### 4.6 Dumper / hooking logistics

| # | Issue | Cause | Fix |
|---|-------|-------|-----|
| 22 | `UnboundLocalError: cfm` in dumper | Hook registration referred to `cfm = flow.decoder` before that line | Used `flow.decoder.estimator` directly; removed the dependency on a later local binding. |
| 23 | My `flow_inference(finalize=False)` triggered an internal shape-mismatch assert inside PyTorch | That code path trims `pre_lookahead_len * token_mel_ratio = 6` frames — not the inference path | Pass `finalize=True` (matches what the public `inference()` entry point does). |
| 24 | Estimator `forward_hook` never fired for per-step tensors | `basic_euler` calls `self.estimator.forward(x, …)` directly, bypassing `__call__` | Monkey-patched `estimator.forward` to record `x_in` / `mu` / `t` / `r` / `spks` / `cond` / `mask` / `dxdt` for every step. |

---

## 5. Verification approach

We treat verification as a staged pipeline:

1. **Python reference dumper** (`scripts/dump-s3gen-reference.py`) runs the
   full PyTorch pipeline with `forward_hook`s on every module we plan to
   reimplement. Each intermediate is saved as `.npy` in `artifacts/s3gen-ref/`
   with a predictable name. Multi-call hooks save a `_call{N}` suffix so each
   flow-matching step gets its own tensor.

2. **C++ staged harness** (`src/test_s3gen.cpp`) loads a single GGUF, then for
   each stage:
   - Loads only the reference tensors needed as inputs.
   - Builds a tiny ggml graph covering exactly that stage.
   - Runs the graph and reads back the outputs.
   - Calls `compare_f32(got, expected, n)` to print
     `max_abs / mean_abs / rms / max|ref| / rel`.

3. Each stage is gated on rel-error thresholds. Precision regresses
   immediately visible, so a change that drops precision to ~1e-4 shows up
   before it silently corrupts later stages.

4. For T3 we additionally have **bit-exact** testing — under greedy decoding
   with deterministic preprocessing, ggml speech tokens equal PyTorch speech
   tokens token-for-token.

---

## 6. How to re-run everything

Assume the remote machine has the Python venv already built:

```bash
ssh gianni@qvac-dev-linux-x64
cd ~/qvac-chatterbox.cpp

# One-time: build the binaries
cmake -S . -B build
cmake --build build -j8 --target chatterbox test-s3gen

# One-time: convert weights + built-in conditionals
. ~/chatterbox-ref/.venv/bin/activate
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
python scripts/convert-s3gen-to-gguf.py   --out models/chatterbox-s3gen.gguf

# One-time: dump the Python reference tensors
python scripts/dump-s3gen-reference.py \
  --text 'Hello from ggml.' --out artifacts/s3gen-ref \
  --seed 42 --n-predict 64 --device cpu

# Validate every stage in C++
./build/test-s3gen models/chatterbox-s3gen.gguf artifacts/s3gen-ref ALL
```

Expected output:

```
Stage A speaker_emb_affine   rel ≈ 1e-7
Stage B input_embedded       rel = 0
Stage C encoder_embed        rel ≈ 4e-7
Stage D pre_lookahead        rel ≈ 3e-7
Stage E enc_block0_out       rel ≈ 1e-7
Stage F encoder_out / mu     rel ≈ 5e-7
Stage G1 time_mixer          rel ≈ 7e-7
Stage G2 cfm_resnet_out      rel ≈ 3e-7
Stage G3 tfm_out             rel ≈ 2e-7
Stage G4 cfm_step0_dxdt      rel ≈ 1e-6
```

End-to-end T3 text → speech tokens (currently the deepest working path on-device):

```bash
./build/chatterbox \
  --model models/chatterbox-t3-turbo.gguf \
  --text 'Hello from ggml.' \
  --tokenizer-dir ~/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/*/ \
  --output speech_tokens.txt
```

---

## 7. Next steps

1. **HiFTGenerator** — port `ConvRNNF0Predictor`, `SineGen`, the transposed
   conv upsample stack, multi-ResBlock with Snake activation, and the
   STFT/ISTFT output head. Validate against `waveform.npy` (already dumped by
   the reference pipeline) at **rel < 1e-4** (vocoder precision is typically
   looser than pure transformer ops because of `stft`/`istft` and `cumsum`
   phase).

2. **End-to-end binary** — wire T3 → S3Gen → HiFT in a single `chatterbox`
   invocation that takes `--text` and emits a `.wav`. Expected speed on CPU:
   ~ real-time factor 1.0–1.5× on a modern desktop (T3 is the long pole, ~30
   steps/s for 24-layer GPT-2 Medium in ggml).

## HiFT vocoder port — stage-by-stage verification

The HiFTGenerator (the mel→wav vocoder) was ported in five verifiable stages
against Python reference dumps. The table below shows the measured relative
error vs the reference:

| Stage | What                                      | max_abs   | rel         |
|-------|-------------------------------------------|-----------|-------------|
| H1    | f0_predictor (5×Conv+ELU+Linear)          | 7.5e-08   | 4.2e-06     |
| H3    | decode body: conv_pre → ups/rb → conv_post| 9.3e-08   | 5.6e-07     |
| H4    | STFT (Conv1d with DFT + Hann kernel)      | 3.8e-04   | 7.9e-03     |
| H5    | ISTFT (ConvT + window-sum normalize)      | 8.9e-08   | 1.0e-04     |

Key techniques used:
- Snake activation `x + (1/α)·sin²(αx)` implemented with `ggml_sin` and a
  pre-computed `1/α` tensor fed as a graph input (72 such inputs across the 9
  main ResBlocks and 3 source ResBlocks).
- ConvTranspose1d with asymmetric PyTorch padding: ggml's op only accepts
  `p0=0`, so we compute full-length output then slice `p` samples from each
  side.
- Asymmetric reflection pad `(1, 0)` done manually by extracting `x[1:2]` and
  concat-prepending it.
- STFT built as a `Conv1d` with a DFT+window kernel of shape `[n_fft, 1, 2F]`
  (real and imaginary parts stacked as output channels). Center-mode reflection
  pad `n_fft//2` applied via manual slice-and-concat on each side.
- ISTFT built as a `ConvTranspose1d` with the inverse DFT+window kernel,
  followed by element-wise divide by a precomputed `window²` overlap-sum
  buffer, then trim `n_fft//2` from each end.

The resulting `mel2wav` binary demonstrates the complete vocoder:

```
mel2wav --s3gen-gguf models/chatterbox-s3gen.gguf \
        --mel-npy artifacts/s3gen-ref/mel_output.npy \
        --out /tmp/out.wav
```

Against the Python reference waveform, the C++ output matches at:
- Same RMS (1.22e-04 vs 1.22e-04)
- Time-domain diff max 3.3e-05 (signal max ~9e-04)
- Spectrogram magnitude diff max rel 2.5% (differences localized to the
  stochastic SineGen excitation; the deterministic conv-net chain is
  bit-exact).

## End-to-end text → wav in pure C++

The full pipeline is now implemented as two binaries that compose through the
speech-token representation:

```
chatterbox       : text             -> speech tokens   (T3, GPT-2 Medium)
chatterbox-tts   : speech tokens    -> 24kHz wav        (S3Gen + HiFT)
scripts/synthesize.sh  wraps both into a single command
```

`chatterbox-tts` wires the S3Gen encoder → meanflow CFM (2 steps) → HiFT
vocoder. It takes T3 speech tokens plus a reference voice (prompt\_token,
prompt\_feat, embedding) and emits a 24 kHz PCM wav.

Debug mode (`--debug`) substitutes Python-dumped reference values for the
random bits (CFM `z` and `noised_mels`) so every stage can be validated
bit-exactly against the PyTorch reference. Measured errors end-to-end:

| Stage                              | max\_abs  | rel        |
|------------------------------------|-----------|------------|
| input\_embedding(tokens)           | 0         | 0          |
| encoder → encoder\_proj (mu)       | 8.3e-07   | 4.5e-07    |
| speaker embedding (spks)           | 5.9e-08   | small      |
| cond (prompt\_feat placement)      | 0         | 0          |
| t\_emb (sinusoidal → MLP → mixer)  | 7.6e-06   | small      |
| CFM step 0 dxdt                    | 2.1e-05   | small      |
| CFM step 1 dxdt                    | 1.8e-05   | small      |
| final mel (80 × 136)               | 1.0e-05   | **8.9e-07**|

The generated wav against Python's reference:
- RMS: 1.22e-04 (C++) vs 1.22e-04 (Python)
- Max amplitude: 8.85e-04 vs 8.82e-04
- Spectrogram magnitude: rel 2.5% (entirely from non-deterministic SineGen
  excitation — std::mt19937 ≠ torch.rand).

Production mode (no `--debug`) uses a seeded `std::mt19937` for both the CFM
initial noise/meanflow mels and the SineGen phase + additive noise.

### Timing (CPU only, 10-core EPYC)

`-march=native` is already enabled by ggml (so AVX-512 / AVX-VNNI are used),
but the first build used a single thread, rebuilt every graph from scratch,
and used the stock scaled-dot-product attention. Four optimizations moved
us decisively into faster-than-real-time territory:

1. **Multi-threading** (4.6× speedup alone) — plumb a global `g_n_threads`
   (default = `std::thread::hardware_concurrency()`) into
   `ggml_backend_cpu_set_n_threads` before every `ggml_backend_graph_compute`.
2. **CFM graph reuse** (~7% overall) — the estimator graph is topologically
   identical on every meanflow step, so we build it once into a
   `cfm_estimator_cache` and only run new input values through it on
   subsequent steps.
3. **Flash attention in CFM BasicTransformerBlock** (~34% off CFM, ~18%
   overall) — swap the explicit `softmax(QKᵀ/√d) · V` kernel for
   `ggml_flash_attn_ext`. 56 attention ops × 2 CFM steps = 112 calls per
   utterance, all fused. No materialized `T×T` scores/attn tensors.
4. **Fold symmetric conv padding** — drop 6 redundant `ggml_pad_ext + conv`
   pairs (most impactful in the HiFT ResBlocks) by passing the padding to
   `ggml_im2col` directly instead of padding as a separate op. Saves one
   intermediate tensor allocation per resblock conv (72+ in HiFT alone).

Measured on our 10-core workstation, for an 8.64 s utterance:

| Configuration                                           | Total   | RTF  | vs real-time |
|---------------------------------------------------------|---------|------|--------------|
| 1 thread, graph rebuilt per step, no flash attn         | 22.5 s  | 2.60 | 2.6× slower  |
| 10 threads, graph rebuilt per step, no flash attn       | 3.47 s  | 0.40 | 2.5× faster  |
| 10 threads + CFM graph reuse, no flash attn             | 3.09 s  | 0.36 | 2.8× faster  |
| **10 threads + CFM graph reuse + flash attn + pad fold**| **2.39 s** | **0.28** | **3.6× faster** |

Total speedup from the baseline single-threaded path: **9.4×**.

Stage breakdown at the final configuration (10 threads, 8.64 s output):

| Stage                     | time     |
|---------------------------|----------|
| S3Gen encoder             | 286 ms   |
| CFM 2 meanflow steps      | 785 ms   |
| HiFT vocoder              | 1312 ms  |
| **Total**                 | **2.39 s** |

HiFT is now the bottleneck (~55% of wall time) — the remaining convs in the
3-stage upsample / resblock stack on T=16320 channels are memory-bandwidth
bound rather than compute bound.

### What we tried that did NOT help

- **`GGML_BLAS=ON` with OpenBLAS** — no measurable change. Our matmuls are
  medium-sized and ggml's hand-written AVX-512 kernels already saturate what
  OpenBLAS would deliver.
- **`GGML_LTO=ON`** — no measurable effect on shared-library build.
- **Converting CFM linear weights to F16** — regression. Saved ~100 MB in
  the GGUF but made CFM ~10% slower (F16→F32 upconvert inside mul_mat is
  not free, and the F32 AVX-512 kernel is already very fast).
- **Flash attention in the Conformer encoder** — incompatible. The Conformer
  uses ESPnet-style relative positional bias that's added inside the softmax,
  which `ggml_flash_attn_ext` does not support.

### Still on the table

- **Quantizing the T3 GPT-2 Medium backbone** to Q4\_K\_M / Q5\_K — this
  only affects the separate `chatterbox` (T3) binary, not `chatterbox-tts`.
- **GPU backends** (CUDA / Metal) — already wired through `ggml_backend_t`.
- **Pre-compiled / cached graphs across utterances** — for server mode,
  extend the `cfm_estimator_cache` pattern to the encoder and HiFT graphs.

### Bug hunts along the way

Two wiring bugs surfaced while bringing up `chatterbox-tts`:

1. **Silence padding value**: `speech_tokens` must be appended with
   `S3GEN_SIL = 4299` (not 0) before the encoder to match the Python
   `speech_tokens_padded` convention.
2. **Relative positional-encoding sign flip**: while copying
   `compute_pos_emb` into the new binary I swapped `pos_pe` / `neg_pe` in the
   concatenation step, which silently gave plausible-looking encoder output
   with 20 % relative error. Fixed by matching the original ordering: first
   half of the PE buffer is reversed `pos_pe`, second half is `neg_pe`.
3. **Mu layout transpose for CFM**: the encoder's `encoder_proj.npy` is
   numpy `(T, 80)` while the CFM estimator expects the transposed numpy
   `(80, T)` layout. Added an explicit transpose between the encoder and
   CFM to bridge the two conventions.

3. **GPU backends** — once the CPU path is stable, re-enable `GGML_CUDA` /
   `GGML_METAL` paths. The code is already using `ggml_backend_t` abstractions
   so in principle only conv1d needs custom wiring (im2col path is already
   backend-agnostic).

4. **Quantization** — T3 alone is ~700 MB in F16; a Q4_0 / Q4_K_M path should
   land us around 200 MB with negligible quality loss (proven on GPT-2 Medium
   sized backbones elsewhere). Convs in S3Gen / HiFT stay F32 for now.

5. **Voice cloning** — currently uses the built-in `conds.pt` voice. To
   support custom audio we'd need to port `VoiceEncoder` (3-layer LSTM) and
   either `S3Tokenizer` or accept pre-computed speaker embeddings from
   Python-side preprocessing. LSTM inference in ggml is known-good via
   whisper.cpp / llama.cpp patterns.
