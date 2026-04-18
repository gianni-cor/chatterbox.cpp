# ggml patches for Chatterbox

`ggml` is vendored as a pristine upstream clone (see the top-level
[`README.md`](../README.md)), so any fixes we need in it live here as
standalone patches and are applied after the clone.

Only the Metal backend currently needs a patch. CPU / CUDA / Vulkan
builds work with stock upstream ggml and can skip this step entirely.

## Apply

```bash
# From the repo root, after cloning upstream ggml into ./ggml.
(cd ggml && git apply ../patches/ggml-metal-chatterbox-ops.patch)

# Then configure + build as usual.  To enable Metal:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

To confirm the patch applied cleanly:

```bash
(cd ggml && git status --short)
# Expected: three modified files under ggml/src/ggml-metal/
```

Skip the `git apply` step on Linux or when building CPU-only.

## `ggml-metal-chatterbox-ops.patch`

Base commit: `58c3805` (`sync : llama.cpp`, 2026-04-09).

Fixes three gaps in ggml-metal that make Chatterbox unusable or very slow
on Metal:

| Symptom                                       | Root cause in ggml-metal                          | What this patch does                                           |
|-----------------------------------------------|---------------------------------------------------|----------------------------------------------------------------|
| T3 crashes: `unsupported op 'DIAG_MASK_INF'`  | No op entry / no kernel                            | Adds `kernel_diag_mask_inf_f32`, dispatcher, `supports_op` case|
| S3Gen crashes: `unsupported op 'PAD'` when any front-pad (`lp0..lp3`) is non-zero | Kernel only supports tail padding; `supports_op` rejects non-zero front pads | Extends `kernel_pad_f32` + `ggml_metal_kargs_pad` to honour `lp0..lp3` and drops the rejection  |
| HiFT decode is ~100× slower than CPU          | `kernel_conv_transpose_1d` is scalar: 1 thread per output pixel iterating over *all* `IC * IL` inputs, with most of the work inside a conditional | Tighten the input-position range to the few that contribute (`i_min..i_max`) and parallelise `IC` across a 32-thread simdgroup with `simd_sum` reduction |
| T3 step does 96 extra `bin_fuse` dispatches/step for `add(bias)` after each Q-variant `mul_mv` | `mul_mv` and the following bias add are separate Metal kernels | Fuse `mul_mat + add(bias)` for the Q-variant mat-vec kernels (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0) via a new `FC_mul_mv_has_bias` function constant and a tiny `helper_mv_add_bias<NR0>` post-pass in each top-level kernel wrapper. The op encoder detects the pattern and returns `n_fuse=2` to consume the ADD |

Measured on M3 Ultra, `hift_decode` at HiFT-realistic shapes:
- Before: ~15 000 ms
- After:    ~350 ms (≈ 40× speedup; end-to-end `gen_RTF` goes from
  unusable → 0.19 on F16)

Correctness is validated against the ggml CPU backend by the
`test-metal-ops` binary built in the parent repo (Metal builds only).
Run it after rebuilding:

```bash
./build/test-metal-ops
# Expected: "diag_mask_inf / pad_ext / conv_transpose_1d: PASS"
```

## Dropping the patch

If upstream ggml merges equivalent fixes, delete the patch file and
remove the `git apply` step from the build instructions.  The C++ side
of Chatterbox uses only ops supported by every backend, so nothing else
needs to change.

No patch is needed for CPU / CUDA / Vulkan — those backends already
handle every op Chatterbox emits.
