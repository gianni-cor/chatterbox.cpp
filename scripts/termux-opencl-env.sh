# Termux / remote OpenCL: use this for real OpenCL + ggml testing.
# macOS is not a supported target (OpenCL is deprecated, headers/loaders
# often break in CMake). Build with -DGGML_OPENCL=ON and run binaries on
# the phone/tablet, or over ssh to that device, not on the local Mac.
#
# Source before the chatterbox binary so libOpenCL and other GPU deps resolve.
#   . "$(dirname "$0")/termux-opencl-env.sh"
#   # or from repo root:
#   . scripts/termux-opencl-env.sh
export LD_LIBRARY_PATH="/data/data/com.termux/files/home/lib:${LD_LIBRARY_PATH:-}"
# Optional: vendor Adreno / Android ICD paths (uncomment if needed)
# export LD_LIBRARY_PATH="/vendor/lib64:/vendor/lib64/egl:/vendor/lib64/hw:${LD_LIBRARY_PATH}"
