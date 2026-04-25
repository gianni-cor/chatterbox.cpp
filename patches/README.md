# ggml patches for Chatterbox

`ggml` is vendored as a pristine upstream clone (see the top-level
[`README.md`](../README.md)), so any fixes we need in it live here as
standalone patches and are applied after the clone.

| Patch | When you need it |
|--------|------------------|
| `ggml-metal-chatterbox-ops.patch` | Building with **Metal** (Apple Silicon T3 + full pipeline). |
| `ggml-opencl-chatterbox-ops.patch` | Building with **OpenCL** (e.g. Android / Termux + Adreno: `CONV_TRANSPOSE_1D` for HiFT, `SIN`, backend notes). |
| (none) | **CPU** / **CUDA** / **Vulkan** only — stock upstream `ggml` is enough. |

`setup-ggml.sh` always applies **both** patches in order (Metal, then
OpenCL).  Extra OpenCL code is inert when you configure without
`GGML_OPENCL=ON`.

## Apply

The top-level [`scripts/setup-ggml.sh`](../scripts/setup-ggml.sh) does
everything for you:

```bash
# From the repo root.  Clones ggml if needed, hard-resets to the pinned
# commit, and applies both patch files.  Re-running overwrites any local
# edits under ./ggml.
./scripts/setup-ggml.sh
```

Then configure + build as usual, for example:

```bash
# Metal (macOS)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build -j$(sysctl -n hw.ncpu)

# OpenCL (e.g. Termux) — set LD_LIBRARY_PATH to your OpenCL/ggml DSOs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

If you'd rather run the steps by hand (e.g. to pin a different upstream
commit), the script is effectively:

```bash
git clone https://github.com/ggml-org/ggml.git ggml
cd ggml && git reset --hard $GGML_COMMIT && git clean -fdq
git apply ../patches/ggml-metal-chatterbox-ops.patch
git apply ../patches/ggml-opencl-chatterbox-ops.patch
```

`GGML_COMMIT` lives at the top of `scripts/setup-ggml.sh` as the
single source of truth — bump it when re-generating patches
against a newer upstream ggml.  To confirm they applied:

```bash
(cd ggml && git status --short)
# Expected: files under src/ggml-metal/, include/ggml-opencl.h, src/ggml-opencl/…
```

Skip `setup-ggml.sh` only if you use `-DTTS_CPP_USE_SYSTEM_GGML=ON` with
another ggml; otherwise the pin + patches keep builds deterministic.

## `ggml-metal-chatterbox-ops.patch`

Base commit: `58c3805` (`sync : llama.cpp`, 2026-04-09).

Fixes three gaps in ggml-metal that make Chatterbox unusable or very slow
on Metal:

| Symptom                                       | Root cause in ggml-metal                          | What this patch does                                           |
|-----------------------------------------------|---------------------------------------------------|----------------------------------------------------------------|
| T3 crashes: `unsupported op 'DIAG_MASK_INF'`  | No op entry / no kernel                            | Adds `kernel_diag_mask_inf_f32`, dispatcher, `supports_op` case|
| S3Gen crashes: `unsupported op 'PAD'` when any front-pad (`lp0..lp3`) is non-zero | Kernel only supports tail padding; `supports_op` rejects non-zero front pads | Extends `kernel_pad_f32` + `ggml_metal_kargs_pad` to honour `lp0..lp3` and drops the rejection  |
| HiFT decode is ~100× slower than CPU          | `kernel_conv_transpose_1d` is scalar: 1 thread per output pixel iterating over *all* `IC * IL` inputs, with most of the work inside a conditional | Tighten the input-position range to the few that contribute (`i_min..i_max`) and parallelise `IC` across a 32-thread simdgroup with `simd_sum` reduction |
| T3 step does `mul_mv + bin_fuse(add)` / `mul_mv + bin_fuse(add+add)` per linear layer | `mul_mv` and the following bias / bias+residual adds are separate Metal kernels even though Vulkan fuses the same patterns (`ggml_vk_can_fuse` + `Fuse0` / `Fuse1` shader bindings) | Fuse `mul_mat + add(bias)` and `mul_mat + add(bias) + add(residual)` for the Q-variant mat-vec kernels (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0) via two function constants (`FC_mul_mv_has_bias`, `FC_mul_mv_has_residual`) and a `helper_mv_add_bias<NR0>` post-pass.  The op encoder tries `{MUL_MAT, ADD, ADD}` first and falls back to `{MUL_MAT, ADD}`; `n_fuse` tells the dispatcher how many nodes to consume |

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

## `ggml-opencl-chatterbox-ops.patch`

Base commit: `58c3805` (same pin as the Metal patch).

Extends `ggml-opencl` so Chatterbox’s S3Gen + **HiFT** path can run on
OpenCL (e.g. Qualcomm Adreno) instead of failing on missing ops:

| What | Purpose |
|------|---------|
| `CONV_TRANSPOSE_1D` | f32 and f16-kernel + f32 input kernels, dispatch + `supports_op` |
| `GGML_OP_SIN` | `sin.cl` (element-wise), dispatch + `supports_op` |
| `ggml-opencl.h` | Document that `ggml_backend_opencl_init` may return NULL when no device |
| Build | Register new `.cl` sources in `CMakeLists.txt` for embed |

Regenerate from a throwaway `ggml` worktree at `GGML_COMMIT` after editing
upstream:

```bash
# From cherry-picked commits or a branch:
(cd ggml && git diff 58c38058..your-branch) > patches/ggml-opencl-chatterbox-ops.patch
# Sanity check on a clean tree:
git -C ggml reset --hard 58c38058 && git -C ggml clean -fdq
git -C ggml apply ../patches/ggml-metal-chatterbox-ops.patch
git -C ggml apply --check ../patches/ggml-opencl-chatterbox-ops.patch
```

## Dropping the patch

If upstream ggml merges equivalent fixes, delete the patch file and
remove the `git apply` step from the build instructions.  The C++ side
of Chatterbox uses only ops supported by every backend, so nothing else
needs to change.

No patch is needed for CPU / CUDA / Vulkan — those backends already
handle every op Chatterbox emits, except where OpenCL still trails;
use this OpenCL patch when targeting OpenCL.
