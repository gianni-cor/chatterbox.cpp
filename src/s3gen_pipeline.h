#pragma once

// Back half of the Chatterbox pipeline: S3Gen encoder → 2-step meanflow CFM →
// HiFT vocoder. Takes T3-generated speech tokens + reference voice conditioning
// and writes a 24 kHz WAV.
//
// Implementation in src/s3gen_pipeline.cpp.

#include <cstdint>
#include <string>
#include <vector>

struct s3gen_synthesize_opts {
    std::string s3gen_gguf_path;  // required: chatterbox-s3gen.gguf
    std::string out_wav_path;     // required: where to write the 24 kHz wav

    // If empty, use the built-in voice embedded in the GGUF
    // (s3gen/builtin/{embedding,prompt_token,prompt_feat}).
    // Otherwise load embedding.npy / prompt_token.npy / prompt_feat.npy from
    // this directory.
    std::string ref_dir;

    // Optional: if non-empty, override the prompt_feat tensor (S3Gen reference
    // mel spectrogram) with these values instead of loading it from
    // ref_dir/prompt_feat.npy or from s3gen/builtin. Layout is row-major
    // (T_mel, 80). Used by --reference-audio in main.cpp to inject a mel
    // computed natively in C++ from a reference wav.
    std::vector<float> prompt_feat_override;
    int prompt_feat_rows_override = 0;

    // Optional: if non-empty, override the 192-d speaker `embedding` that's
    // produced by CAMPPlus.  Same motivation as prompt_feat_override: lets
    // main.cpp replace Python's embedding.npy with a C++ CAMPPlus output
    // when --reference-audio is given.
    std::vector<float> embedding_override;

    // Optional: if non-empty, override the S3Gen-side reference speech
    // tokens (`prompt_token`).  Populated from --reference-audio via
    // S3TokenizerV2 in main.cpp (Phase 2e).
    std::vector<int32_t> prompt_token_override;

    int  seed      = 42;
    int  n_threads = 0;          // 0 = hardware_concurrency
    int  sr        = 24000;
    bool debug     = false;      // validation mode; requires ref_dir

    // When > 0, try to run S3Gen + HiFT on a GPU backend (CUDA / Metal / Vulkan
    // depending on what the build enables).  Falls back to CPU if the backend
    // cannot be initialised.  The actual layer count is not yet used for split
    // offload; any positive value enables the GPU path.
    int  n_gpu_layers = 0;
};

// Runs encoder + CFM + HiFT on the given T3 speech tokens and writes a WAV.
// Returns 0 on success, non-zero on error.
int s3gen_synthesize_to_wav(
    const std::vector<int32_t> & speech_tokens,
    const s3gen_synthesize_opts & opts);
