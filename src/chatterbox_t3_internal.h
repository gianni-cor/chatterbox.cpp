#pragma once

// Library-internal declarations for the T3 (GPT-2 Medium autoregressive
// text -> speech tokens) front-half of the Chatterbox pipeline.  Shared
// between src/main.cpp (CLI) and src/chatterbox_engine.cpp (public engine
// API under include/tts-cpp/chatterbox/engine.h).
//
// This header is NOT installed with the library (see CMakeLists.txt
// install rules) — it's part of the implementation, not the public
// surface.  Public consumers should use <tts-cpp/chatterbox/engine.h>.
//
// Names stay in the global namespace to keep main.cpp's diff minimal
// during the extraction (main.cpp had them as static globals/functions).

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

constexpr int CHBX_MAX_NODES = 8192;

constexpr const char * KEY_TEXT_VOCAB_SIZE   = "chatterbox.text_vocab_size";
constexpr const char * KEY_SPEECH_VOCAB_SIZE = "chatterbox.speech_vocab_size";
constexpr const char * KEY_START_SPEECH      = "chatterbox.start_speech_token";
constexpr const char * KEY_STOP_SPEECH       = "chatterbox.stop_speech_token";
constexpr const char * KEY_SPEAKER_EMBED     = "chatterbox.speaker_embed_size";
constexpr const char * KEY_LAYER_NORM_EPS    = "chatterbox.layer_norm_eps";
constexpr const char * KEY_COND_PROMPT_LEN   = "chatterbox.cond_prompt_length";
constexpr const char * KEY_N_CTX             = "chatterbox.n_ctx";
constexpr const char * KEY_N_EMBD            = "chatterbox.n_embd";
constexpr const char * KEY_N_HEAD            = "chatterbox.n_head";
constexpr const char * KEY_N_LAYER           = "chatterbox.n_layer";

struct chatterbox_hparams {
    int32_t n_text_vocab       = 0;
    int32_t n_speech_vocab     = 0;
    int32_t start_speech_token = 0;
    int32_t stop_speech_token  = 0;
    int32_t n_ctx              = 0;
    int32_t n_embd             = 0;
    int32_t n_head             = 0;
    int32_t n_layer            = 0;
    int32_t speaker_embed_size = 0;
    int32_t cond_prompt_len    = 0;
    float   eps                = 1e-5f;
};

struct gpt2_layer {
    ggml_tensor * ln_1_g = nullptr;
    ggml_tensor * ln_1_b = nullptr;
    ggml_tensor * ln_2_g = nullptr;
    ggml_tensor * ln_2_b = nullptr;

    ggml_tensor * c_attn_attn_w = nullptr;
    ggml_tensor * c_attn_attn_b = nullptr;
    ggml_tensor * c_attn_proj_w = nullptr;
    ggml_tensor * c_attn_proj_b = nullptr;

    ggml_tensor * c_mlp_fc_w   = nullptr;
    ggml_tensor * c_mlp_fc_b   = nullptr;
    ggml_tensor * c_mlp_proj_w = nullptr;
    ggml_tensor * c_mlp_proj_b = nullptr;
};

struct chatterbox_model {
    chatterbox_hparams hparams;

    ggml_tensor * wpe              = nullptr;
    ggml_tensor * ln_f_g           = nullptr;
    ggml_tensor * ln_f_b           = nullptr;
    ggml_tensor * text_emb         = nullptr;
    ggml_tensor * speech_emb       = nullptr;
    ggml_tensor * speech_head      = nullptr;
    ggml_tensor * speech_head_bias = nullptr;
    ggml_tensor * cond_spkr_w      = nullptr;
    ggml_tensor * cond_spkr_b      = nullptr;

    ggml_tensor * builtin_speaker_emb        = nullptr;
    ggml_tensor * builtin_cond_prompt_tokens = nullptr;

    std::vector<gpt2_layer> layers;

    ggml_tensor * memory_k = nullptr;
    ggml_tensor * memory_v = nullptr;

    ggml_context * ctx_w  = nullptr;
    ggml_context * ctx_kv = nullptr;

    ggml_backend_t backend = nullptr;

    ggml_backend_buffer_t buffer_w  = nullptr;
    ggml_backend_buffer_t buffer_kv = nullptr;

    ggml_context *        ctx_override    = nullptr;
    ggml_backend_buffer_t buffer_override = nullptr;

    std::map<std::string, ggml_tensor *> tensors;

    std::vector<std::string> tok_tokens;
    std::vector<std::string> tok_merges;
};

struct chatterbox_sampling_params {
    int32_t top_k          = 1000;
    float   top_p          = 0.95f;
    float   temp           = 0.8f;
    float   repeat_penalty = 1.2f;
};

ggml_backend_t init_backend(int n_gpu_layers);

bool load_model_gguf(
    const std::string & path,
    chatterbox_model &  model,
    int                 requested_ctx,
    int                 n_gpu_layers);

bool eval_prompt(
    const chatterbox_model &     model,
    ggml_gallocr_t               allocr,
    int                          n_threads,
    const std::vector<int32_t> & text_tokens,
    std::vector<float> &         logits_out,
    int &                        prompt_len);

bool eval_step(
    const chatterbox_model & model,
    ggml_gallocr_t           allocr,
    int                      n_threads,
    int                      n_past,
    int32_t                  token,
    std::vector<float> &     logits_out);

int32_t sample_next_token_ex(
    const std::vector<float> &          logits,
    const std::vector<int32_t> &        generated,
    const chatterbox_sampling_params &  params,
    std::mt19937 &                      rng);

void chatterbox_log_cb(ggml_log_level level, const char * text, void * ud);

extern int g_log_verbose;

bool compute_prompt_feat_native(
    const std::string &  wav_path,
    const std::string &  s3gen_gguf,
    std::vector<float> & prompt_feat,
    int &                prompt_feat_rows,
    bool                 verbose);

bool compute_embedding_native(
    const std::string &  wav_path,
    const std::string &  s3gen_gguf,
    std::vector<float> & embedding,
    ggml_backend_t       backend,
    bool                 verbose);

bool compute_speech_tokens_native(
    const std::string &    wav_path,
    const std::string &    s3gen_gguf,
    int                    max_cond_tokens,
    std::vector<int32_t> & prompt_token,
    std::vector<int32_t> & cond_prompt_tokens,
    int                    n_threads,
    ggml_backend_t         backend,
    bool                   verbose);

bool validate_reference_audio(const std::string & path);
