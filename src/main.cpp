#include "gpt2_bpe.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "s3gen_pipeline.h"
#include "npy.h"
#include "voice_features.h"
#include "voice_encoder.h"
#include "campplus.h"
#include "s3tokenizer.h"
#include "gguf.h"

#include <sys/stat.h>

static bool file_exists(const std::string & path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Load REF.wav, resample to 24 kHz if needed, pull the 80-ch mel filterbank out
// of the s3gen GGUF, and compute prompt_feat (log-mel) in C++.  out_rows is the
// number of mel frames (= T_mel in the row-major (T_mel, 80) layout).
static bool compute_prompt_feat_native(const std::string & wav_path,
                                       const std::string & s3gen_gguf_path,
                                       std::vector<float> & out_feat,
                                       int & out_rows)
{
    fprintf(stderr, "voice: loading %s\n", wav_path.c_str());
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    fprintf(stderr, "voice:   sr=%d samples=%zu (%.2f s)\n", sr, wav.size(), (double)wav.size() / sr);
    if (sr != 24000) {
        fprintf(stderr, "voice: resampling %d -> 24000\n", sr);
        wav = resample_sinc(wav, sr, 24000);
    }

    // Pull the precomputed mel filterbank out of chatterbox-s3gen.gguf.
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(s3gen_gguf_path.c_str(), gp);
    if (!g) {
        fprintf(stderr, "voice: failed to open %s\n", s3gen_gguf_path.c_str());
        return false;
    }
    ggml_tensor * fb = ggml_get_tensor(tmp_ctx, "s3gen/mel_fb/24k_80");
    if (!fb) {
        fprintf(stderr, "voice: s3gen/mel_fb/24k_80 missing from GGUF; re-run convert-s3gen-to-gguf.py\n");
        gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    std::vector<float> mel_fb(ggml_nelements(fb));
    std::memcpy(mel_fb.data(), ggml_get_data(fb), ggml_nbytes(fb));
    gguf_free(g);
    if (tmp_ctx) ggml_free(tmp_ctx);

    out_feat = mel_extract_24k_80(wav, mel_fb);
    if (out_feat.empty()) return false;
    out_rows = (int)(out_feat.size() / 80);
    fprintf(stderr, "voice: prompt_feat shape=(%d, 80)\n", out_rows);
    return true;
}

// Compute the 192-d `embedding` tensor natively by running the reference wav
// through the C++ Kaldi fbank → mean-subtract → CAMPPlus pipeline.  Returns
// false silently if the s3gen GGUF predates Phase 2d-a (no CAMPPlus tensors).
static bool compute_embedding_native(const std::string & wav_path,
                                     const std::string & s3gen_gguf_path,
                                     std::vector<float> & out_emb)
{
    campplus_weights w;
    if (!campplus_load(s3gen_gguf_path, w)) {
        fprintf(stderr, "voice: s3gen GGUF has no CAMPPlus weights; cannot synthesise "
                        "embedding natively (re-run convert-s3gen-to-gguf.py)\n");
        return false;
    }

    // Mel filterbank for the Kaldi-style fbank lives alongside CAMPPlus in
    // the s3gen GGUF (baked in by the converter).
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(s3gen_gguf_path.c_str(), gp);
    if (!g) return false;
    ggml_tensor * fb_t = ggml_get_tensor(tmp_ctx, "campplus/mel_fb_kaldi_80");
    if (!fb_t) {
        fprintf(stderr, "voice: campplus/mel_fb_kaldi_80 missing; rerun converter\n");
        gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    std::vector<float> mel_fb(ggml_nelements(fb_t));
    std::memcpy(mel_fb.data(), ggml_get_data(fb_t), ggml_nbytes(fb_t));
    gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);

    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);

    std::vector<float> fbank = fbank_kaldi_80(wav, mel_fb);
    if (fbank.empty()) return false;
    const int T = (int)(fbank.size() / 80);

    // Per-utterance mean subtraction over T (matches extract_feature()).
    std::vector<float> col_mean(80, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) col_mean[c] += fbank[(size_t)t * 80 + c];
    for (int c = 0; c < 80; ++c) col_mean[c] /= (float)T;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) fbank[(size_t)t * 80 + c] -= col_mean[c];

    if (!campplus_embed(fbank, T, w, out_emb)) return false;
    fprintf(stderr, "voice: embedding shape=(%zu,) via CAMPPlus (%d fbank frames)\n",
            out_emb.size(), T);
    return true;
}

// Tokenize a reference wav with S3TokenizerV2 (the C++ port of the 25 Hz
// speech-token encoder).  Produces both the S3Gen-side prompt_token stream
// (first 10 s → up to 250 tokens) and the T3-side cond_prompt_speech_tokens
// stream (first 15 s → up to max_cond_tokens tokens).
//
// Returns false if the s3gen GGUF pre-dates Phase 2e (no s3tokv2.* tensors).
static bool compute_speech_tokens_native(const std::string & wav_path,
                                         const std::string & s3gen_gguf_path,
                                         int max_cond_tokens,
                                         std::vector<int32_t> & out_prompt_tokens,
                                         std::vector<int32_t> & out_cond_tokens,
                                         int n_threads)
{
    s3tokv2_weights w;
    if (!s3tokv2_load(s3gen_gguf_path, w)) {
        fprintf(stderr, "voice: s3gen GGUF has no S3TokenizerV2 weights; cannot "
                        "synthesise speech tokens natively (re-run converter)\n");
        return false;
    }

    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);

    // prompt_token: first 10 s of the reference → up to 250 tokens (S3Gen side).
    const int dec_cond_samples = 10 * 16000;
    std::vector<float> prompt_wav(wav.begin(), wav.begin() + std::min((int)wav.size(), dec_cond_samples));
    if (!s3tokv2_tokenize(prompt_wav, w, /*max_tokens=*/-1, out_prompt_tokens, n_threads)) return false;

    // cond_prompt_speech_tokens: first 15 s → up to max_cond_tokens (T3 side).
    const int enc_cond_samples = 15 * 16000;
    std::vector<float> cond_wav(wav.begin(), wav.begin() + std::min((int)wav.size(), enc_cond_samples));
    if (!s3tokv2_tokenize(cond_wav, w, max_cond_tokens, out_cond_tokens, n_threads)) return false;

    fprintf(stderr, "voice: prompt_token=(%zu,) cond_prompt_speech_tokens=(%zu,) via S3TokenizerV2\n",
            out_prompt_tokens.size(), out_cond_tokens.size());
    return true;
}

static constexpr int CHBX_MAX_NODES = 8192;

// --------------------------------------------------------------------------
// GGUF metadata keys
// --------------------------------------------------------------------------

static constexpr const char * KEY_TEXT_VOCAB_SIZE   = "chatterbox.text_vocab_size";
static constexpr const char * KEY_SPEECH_VOCAB_SIZE = "chatterbox.speech_vocab_size";
static constexpr const char * KEY_START_SPEECH      = "chatterbox.start_speech_token";
static constexpr const char * KEY_STOP_SPEECH       = "chatterbox.stop_speech_token";
static constexpr const char * KEY_SPEAKER_EMBED     = "chatterbox.speaker_embed_size";
static constexpr const char * KEY_LAYER_NORM_EPS    = "chatterbox.layer_norm_eps";
static constexpr const char * KEY_COND_PROMPT_LEN   = "chatterbox.cond_prompt_length";
static constexpr const char * KEY_N_CTX             = "chatterbox.n_ctx";
static constexpr const char * KEY_N_EMBD            = "chatterbox.n_embd";
static constexpr const char * KEY_N_HEAD            = "chatterbox.n_head";
static constexpr const char * KEY_N_LAYER           = "chatterbox.n_layer";

// --------------------------------------------------------------------------
// Model structs
// --------------------------------------------------------------------------

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

    // Override buffer: populated only when --ref-dir supplies a cond_prompt
    // tensor whose length differs from the GGUF's built-in. The original
    // built-in tensor stays alive (reachable via ctx_w) but unused.
    ggml_context *        ctx_override    = nullptr;
    ggml_backend_buffer_t buffer_override = nullptr;

    std::map<std::string, ggml_tensor *> tensors;

    // GPT-2 BPE tokenizer, carried in the GGUF as tokenizer.ggml.* metadata.
    std::vector<std::string> tok_tokens;
    std::vector<std::string> tok_merges;
};

// --------------------------------------------------------------------------
// CLI
// --------------------------------------------------------------------------

struct cli_params {
    std::string model;           // T3 GGUF (required unless --tokens-file + --s3gen-gguf)
    std::string tokens_file;     // optional pre-tokenized speech tokens (skips T3)
    std::string text;            // input text for T3
    std::string output;          // legacy: speech-tokens output file (if set, write tokens)
    // S3Gen + HiFT vocoder:
    std::string s3gen_gguf;      // enables full text → wav pipeline
    std::string out_wav;         // wav output path (requires --s3gen-gguf)
    std::string ref_dir;         // override built-in voice with .npy reference dump
    std::string reference_audio; // wav file; computes prompt_feat natively in C++
    bool    debug          = false;  // --debug: load Python-dumped intermediates for validation
    bool    dump_tokens_only = false;
    int32_t seed           = 0;
    int32_t n_threads      = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict      = 1000;   // matches Python's default-ish output budget for paragraph-length text
    int32_t n_ctx          = 0;
    int32_t n_gpu_layers   = 0;
    // Sampling defaults matched to ChatterboxTurboTTS.generate() in tts_turbo.py:
    //   temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2
    // The previous greedy defaults (top_k=1) collapse into silence-token
    // repetition loops on any non-trivial text.
    int32_t top_k          = 1000;
    float   top_p          = 0.95f;
    float   temp           = 0.8f;
    float   repeat_penalty = 1.2f;
};

static void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s --model MODEL.gguf [--text TEXT | --tokens-file tokens.txt] [options]\n", argv0);
    fprintf(stderr, "\noptions:\n");
    fprintf(stderr, "  --model PATH            GGUF model produced by convert-t3-turbo-to-gguf.py\n");
    fprintf(stderr, "                          (must embed tokenizer.ggml.* metadata; produced by the\n");
    fprintf(stderr, "                          current converter)\n");
    fprintf(stderr, "  --text TEXT             Input text (uses the GPT-2 BPE tokenizer embedded in GGUF)\n");
    fprintf(stderr, "  --tokens-file PATH      Pre-tokenized text token ids (alternative to --text).\n");
    fprintf(stderr, "                          With --s3gen-gguf this is interpreted as *speech* tokens\n");
    fprintf(stderr, "                          and the T3 step is skipped.\n");
    fprintf(stderr, "  --output PATH           Write generated speech tokens to PATH (text mode).\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  --s3gen-gguf PATH       Enables the full text -> wav pipeline (S3Gen + HiFT).\n");
    fprintf(stderr, "  --out PATH              Output wav file when --s3gen-gguf is set.\n");
    fprintf(stderr, "  --ref-dir DIR           Override built-in voice with embedding.npy /\n");
    fprintf(stderr, "                          prompt_token.npy / prompt_feat.npy from DIR, plus\n");
    fprintf(stderr, "                          T3 speaker_emb.npy / cond_prompt_speech_tokens.npy.\n");
    fprintf(stderr, "  --reference-audio PATH  Reference .wav; prompt_feat is computed natively in\n");
    fprintf(stderr, "                          C++ (replaces ref-dir/prompt_feat.npy). The other\n");
    fprintf(stderr, "                          four voice tensors still come from --ref-dir.\n");
    fprintf(stderr, "  --debug                 Load reference intermediates from --ref-dir for\n");
    fprintf(stderr, "                          bit-exact numerical validation (requires --ref-dir).\n");
    fprintf(stderr, "  --seed N                RNG seed (default: 0)\n");
    fprintf(stderr, "  --threads N             CPU threads (default: %d)\n", std::min(4, (int32_t) std::thread::hardware_concurrency()));
    fprintf(stderr, "  --n-predict N           Max speech tokens (default: 1000)\n");
    fprintf(stderr, "  --context N             Override KV context length\n");
    fprintf(stderr, "  --n-gpu-layers N        GPU backend when N > 0\n");
    fprintf(stderr, "  --top-k N               (default: 1000, matches Python; use 1 for greedy)\n");
    fprintf(stderr, "  --top-p P               (default: 0.95)\n");
    fprintf(stderr, "  --temp T                (default: 0.8)\n");
    fprintf(stderr, "  --repeat-penalty R      (default: 1.2)\n");
    fprintf(stderr, "  -h, --help\n");
}

static bool parse_args(int argc, char ** argv, cli_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s requires an argument\n", flag); return nullptr; }
            return argv[++i];
        };

        if      (arg == "--model")          { auto v = next("--model");          if (!v) return false; params.model = v; }
        else if (arg == "--text")           { auto v = next("--text");           if (!v) return false; params.text = v; }
        else if (arg == "--tokens-file")    { auto v = next("--tokens-file");    if (!v) return false; params.tokens_file = v; }
        else if (arg == "--output")         { auto v = next("--output");         if (!v) return false; params.output = v; }
        else if (arg == "--s3gen-gguf")     { auto v = next("--s3gen-gguf");     if (!v) return false; params.s3gen_gguf = v; }
        else if (arg == "--out")            { auto v = next("--out");            if (!v) return false; params.out_wav = v; }
        else if (arg == "--ref-dir")        { auto v = next("--ref-dir");        if (!v) return false; params.ref_dir = v; }
        else if (arg == "--reference-audio"){ auto v = next("--reference-audio");if (!v) return false; params.reference_audio = v; }
        else if (arg == "--debug")          { params.debug = true; }
        else if (arg == "--seed")           { auto v = next("--seed");           if (!v) return false; params.seed = std::stoi(v); }
        else if (arg == "--threads")        { auto v = next("--threads");        if (!v) return false; params.n_threads = std::stoi(v); }
        else if (arg == "--n-predict")      { auto v = next("--n-predict");      if (!v) return false; params.n_predict = std::stoi(v); }
        else if (arg == "--context")        { auto v = next("--context");        if (!v) return false; params.n_ctx = std::stoi(v); }
        else if (arg == "--n-gpu-layers")   { auto v = next("--n-gpu-layers");   if (!v) return false; params.n_gpu_layers = std::stoi(v); }
        else if (arg == "--top-k")          { auto v = next("--top-k");          if (!v) return false; params.top_k = std::stoi(v); }
        else if (arg == "--top-p")          { auto v = next("--top-p");          if (!v) return false; params.top_p = std::stof(v); }
        else if (arg == "--temp")           { auto v = next("--temp");           if (!v) return false; params.temp = std::stof(v); }
        else if (arg == "--repeat-penalty") { auto v = next("--repeat-penalty"); if (!v) return false; params.repeat_penalty = std::stof(v); }
        else if (arg == "--dump-tokens-only") { params.dump_tokens_only = true; }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); std::exit(0); }
        else { fprintf(stderr, "error: unknown argument: %s\n", arg.c_str()); return false; }
    }
    if (params.dump_tokens_only) {
        if (params.text.empty()) {
            fprintf(stderr, "error: --dump-tokens-only requires --text\n");
            return false;
        }
        return true;
    }
    // If we're only doing the S3Gen+HiFT back half (user already has speech tokens),
    // --model (T3) is optional; otherwise it's required.
    const bool skip_t3 = !params.s3gen_gguf.empty() && !params.tokens_file.empty() && params.text.empty();
    if (!skip_t3 && params.model.empty()) {
        fprintf(stderr, "error: --model is required (pass --s3gen-gguf + --tokens-file to skip T3)\n");
        return false;
    }
    if (params.text.empty() && params.tokens_file.empty()) {
        fprintf(stderr, "error: either --text or --tokens-file is required\n"); return false;
    }
    if (!params.s3gen_gguf.empty() && params.out_wav.empty()) {
        fprintf(stderr, "error: --s3gen-gguf requires --out PATH.wav\n"); return false;
    }
    if (params.debug && params.ref_dir.empty()) {
        fprintf(stderr, "error: --debug requires --ref-dir\n"); return false;
    }
    return true;
}

// --------------------------------------------------------------------------
// I/O helpers
// --------------------------------------------------------------------------

static std::vector<int32_t> read_token_file(const std::string & path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("failed to open token file: " + path);
    std::string raw((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    for (char & ch : raw) if (ch == ',') ch = ' ';
    std::vector<int32_t> tokens;
    std::stringstream ss(raw);
    int32_t tok;
    while (ss >> tok) tokens.push_back(tok);
    return tokens;
}

static void write_token_file(const std::string & path, const std::vector<int32_t> & tokens) {
    std::ofstream fout(path);
    if (!fout) throw std::runtime_error("failed to open output file: " + path);
    for (size_t i = 0; i < tokens.size(); ++i) { if (i) fout << ','; fout << tokens[i]; }
    fout << '\n';
}

// --------------------------------------------------------------------------
// GGUF helpers
// --------------------------------------------------------------------------

static int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}

static ggml_tensor * require_tensor(const chatterbox_model & m, const char * name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end() || !it->second) throw std::runtime_error(std::string("missing tensor: ") + name);
    return it->second;
}

// --------------------------------------------------------------------------
// Backend init
// --------------------------------------------------------------------------

static ggml_backend_t init_backend(int n_gpu_layers) {
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        auto * b = ggml_backend_cuda_init(0);
        if (b) { fprintf(stderr, "%s: using CUDA backend\n", __func__); return b; }
    }
#endif
#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        auto * b = ggml_backend_metal_init();
        if (b) { fprintf(stderr, "%s: using Metal backend\n", __func__); return b; }
    }
#endif
    auto * b = ggml_backend_cpu_init();
    if (!b) throw std::runtime_error("ggml_backend_cpu_init() failed");
    fprintf(stderr, "%s: using CPU backend\n", __func__);
    return b;
}

// --------------------------------------------------------------------------
// Model loading
// --------------------------------------------------------------------------

static bool load_model_gguf(const std::string & path, chatterbox_model & model, int requested_ctx, int n_gpu_layers) {
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gguf_params = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) { fprintf(stderr, "%s: failed to open '%s'\n", __func__, path.c_str()); return false; }

    try {
        auto & hp = model.hparams;
        hp.n_text_vocab       = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_TEXT_VOCAB_SIZE));
        hp.n_speech_vocab     = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_SPEECH_VOCAB_SIZE));
        hp.start_speech_token = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_START_SPEECH));
        hp.stop_speech_token  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_STOP_SPEECH));
        hp.speaker_embed_size = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_SPEAKER_EMBED));
        hp.cond_prompt_len    = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_COND_PROMPT_LEN));
        hp.eps                = gguf_get_val_f32(gguf_ctx, require_key(gguf_ctx, KEY_LAYER_NORM_EPS));
        hp.n_ctx   = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_CTX));
        hp.n_embd  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_EMBD));
        hp.n_head  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_HEAD));
        hp.n_layer = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_LAYER));
        if (requested_ctx > 0) hp.n_ctx = std::min(hp.n_ctx, requested_ctx);

        model.backend = init_backend(n_gpu_layers);

        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        ggml_init_params params = { ggml_tensor_overhead() * (size_t) num_tensors, nullptr, true };
        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) throw std::runtime_error("ggml_init() failed");

        for (int64_t i = 0; i < num_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
            ggml_tensor * dst = ggml_dup_tensor(model.ctx_w, src);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;
        }
        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w); cur; cur = ggml_get_next_tensor(model.ctx_w, cur)) {
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
            ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
        }

        model.wpe              = require_tensor(model, "model/wpe");
        model.ln_f_g           = require_tensor(model, "model/ln_f/g");
        model.ln_f_b           = require_tensor(model, "model/ln_f/b");
        model.text_emb         = require_tensor(model, "chatterbox/text_emb");
        model.speech_emb       = require_tensor(model, "chatterbox/speech_emb");
        model.speech_head      = require_tensor(model, "chatterbox/speech_head");
        model.speech_head_bias = require_tensor(model, "chatterbox/speech_head_bias");
        model.cond_spkr_w      = require_tensor(model, "chatterbox/cond_spkr/w");
        model.cond_spkr_b      = require_tensor(model, "chatterbox/cond_spkr/b");
        model.builtin_speaker_emb        = require_tensor(model, "chatterbox/builtin/speaker_emb");
        model.builtin_cond_prompt_tokens = require_tensor(model, "chatterbox/builtin/cond_prompt_speech_tokens");

        model.layers.resize(hp.n_layer);
        for (int i = 0; i < hp.n_layer; ++i) {
            auto & l = model.layers[i];
            std::string p = "model/h" + std::to_string(i);
            l.ln_1_g        = require_tensor(model, (p + "/ln_1/g").c_str());
            l.ln_1_b        = require_tensor(model, (p + "/ln_1/b").c_str());
            l.ln_2_g        = require_tensor(model, (p + "/ln_2/g").c_str());
            l.ln_2_b        = require_tensor(model, (p + "/ln_2/b").c_str());
            l.c_attn_attn_w = require_tensor(model, (p + "/attn/c_attn/w").c_str());
            l.c_attn_attn_b = require_tensor(model, (p + "/attn/c_attn/b").c_str());
            l.c_attn_proj_w = require_tensor(model, (p + "/attn/c_proj/w").c_str());
            l.c_attn_proj_b = require_tensor(model, (p + "/attn/c_proj/b").c_str());
            l.c_mlp_fc_w    = require_tensor(model, (p + "/mlp/c_fc/w").c_str());
            l.c_mlp_fc_b    = require_tensor(model, (p + "/mlp/c_fc/b").c_str());
            l.c_mlp_proj_w  = require_tensor(model, (p + "/mlp/c_proj/w").c_str());
            l.c_mlp_proj_b  = require_tensor(model, (p + "/mlp/c_proj/b").c_str());
        }

        ggml_init_params kv_params = { ggml_tensor_overhead() * 2, nullptr, true };
        model.ctx_kv = ggml_init(kv_params);
        int64_t n_elements = (int64_t) hp.n_embd * hp.n_layer * hp.n_ctx;
        model.memory_k = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, n_elements);
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);

        fprintf(stderr, "%s: ctx=%d embd=%d layers=%d heads=%d text_vocab=%d speech_vocab=%d cond_prompt=%d\n",
                __func__, hp.n_ctx, hp.n_embd, hp.n_layer, hp.n_head,
                hp.n_text_vocab, hp.n_speech_vocab, hp.cond_prompt_len);
        fprintf(stderr, "%s: weights=%.2f MB  KV=%.2f MB\n", __func__,
                ggml_backend_buffer_get_size(model.buffer_w) / (1024.0*1024.0),
                ggml_backend_buffer_get_size(model.buffer_kv) / (1024.0*1024.0));

        // Read embedded tokenizer arrays if present (added by converter-v2).
        {
            const int64_t tok_kid = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
            const int64_t mer_kid = gguf_find_key(gguf_ctx, "tokenizer.ggml.merges");
            if (tok_kid >= 0 && mer_kid >= 0) {
                const size_t n_tok = gguf_get_arr_n(gguf_ctx, tok_kid);
                const size_t n_mer = gguf_get_arr_n(gguf_ctx, mer_kid);
                model.tok_tokens.reserve(n_tok);
                for (size_t i = 0; i < n_tok; ++i) {
                    model.tok_tokens.emplace_back(gguf_get_arr_str(gguf_ctx, tok_kid, i));
                }
                model.tok_merges.reserve(n_mer);
                for (size_t i = 0; i < n_mer; ++i) {
                    model.tok_merges.emplace_back(gguf_get_arr_str(gguf_ctx, mer_kid, i));
                }
                fprintf(stderr, "%s: tokenizer embedded (%zu tokens, %zu merges)\n",
                        __func__, n_tok, n_mer);
            } else {
                fprintf(stderr, "%s: no embedded tokenizer; --tokenizer-dir will be required for --text\n",
                        __func__);
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: %s\n", __func__, e.what());
        gguf_free(gguf_ctx); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    return true;
}

// --------------------------------------------------------------------------
// GPT-2 transformer core (shared by prompt and step graphs)
// --------------------------------------------------------------------------

static ggml_tensor * build_transformer_core(
    ggml_context * ctx, ggml_cgraph * gf,
    const chatterbox_model & model,
    ggml_tensor * inpL, int n_past, int N) {

    const auto & hp = model.hparams;
    const int n_embd = hp.n_embd, n_head = hp.n_head, n_layer = hp.n_layer, n_ctx = hp.n_ctx;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * cur;
        cur = ggml_norm(ctx, inpL, hp.eps);
        cur = ggml_add(ctx, ggml_mul(ctx, cur, model.layers[il].ln_1_g), model.layers[il].ln_1_b);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_attn_attn_w, cur), model.layers[il].c_attn_attn_b);

        ggml_tensor * Qcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
        ggml_tensor * Kcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
        ggml_tensor * Vcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

        {
            ggml_tensor * k = ggml_view_1d(ctx, model.memory_k, (int64_t)N*n_embd,
                (size_t)ggml_element_size(model.memory_k)*n_embd*((size_t)il*n_ctx+n_past));
            ggml_tensor * v = ggml_view_1d(ctx, model.memory_v, (int64_t)N*n_embd,
                (size_t)ggml_element_size(model.memory_v)*n_embd*((size_t)il*n_ctx+n_past));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v));
        }

        ggml_tensor * Q = ggml_permute(ctx, ggml_cont_3d(ctx, Qcur, n_embd/n_head, n_head, N), 0,2,1,3);
        ggml_tensor * K = ggml_permute(ctx,
            ggml_reshape_3d(ctx,
                ggml_view_1d(ctx, model.memory_k, (int64_t)(n_past+N)*n_embd,
                    (size_t)il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                n_embd/n_head, n_head, n_past+N),
            0,2,1,3);

        ggml_tensor * KQ = ggml_soft_max(ctx,
            ggml_diag_mask_inf(ctx,
                ggml_scale(ctx, ggml_mul_mat(ctx, K, Q), 1.0f/std::sqrt((float)n_embd/n_head)),
                n_past));

        ggml_tensor * V_trans = ggml_cont_3d(ctx,
            ggml_permute(ctx,
                ggml_reshape_3d(ctx,
                    ggml_view_1d(ctx, model.memory_v, (int64_t)(n_past+N)*n_embd,
                        (size_t)il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                    n_embd/n_head, n_head, n_past+N),
                1,2,0,3),
            n_past+N, n_embd/n_head, n_head);

        cur = ggml_cont_2d(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, V_trans, KQ), 0,2,1,3), n_embd, N);
        cur = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_attn_proj_w, cur), model.layers[il].c_attn_proj_b), inpL);

        ggml_tensor * inpFF = cur;
        cur = ggml_norm(ctx, inpFF, hp.eps);
        cur = ggml_add(ctx, ggml_mul(ctx, cur, model.layers[il].ln_2_g), model.layers[il].ln_2_b);
        cur = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_mlp_fc_w, cur), model.layers[il].c_mlp_fc_b));
        cur = ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_mlp_proj_w, cur), model.layers[il].c_mlp_proj_b);

        inpL = ggml_add(ctx, cur, inpFF);
    }

    inpL = ggml_norm(ctx, inpL, hp.eps);
    inpL = ggml_add(ctx, ggml_mul(ctx, inpL, model.ln_f_g), model.ln_f_b);

    ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, model.speech_head, inpL), model.speech_head_bias);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return logits;
}

// --------------------------------------------------------------------------
// Graph builders
// --------------------------------------------------------------------------

static ggml_cgraph * build_prompt_graph(const chatterbox_model & model, int n_text_tokens) {
    const int N = 1 + model.hparams.cond_prompt_len + n_text_tokens + 1;
    static size_t buf_size = ggml_tensor_overhead()*CHBX_MAX_NODES + ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_tokens, "text_tokens"); ggml_set_input(text_tokens);
    ggml_tensor * start_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(start_token, "speech_token"); ggml_set_input(start_token);
    ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(position, "position"); ggml_set_input(position);

    ggml_tensor * spkr = ggml_add(ctx, ggml_mul_mat(ctx, model.cond_spkr_w, model.builtin_speaker_emb), model.cond_spkr_b);
    ggml_tensor * cond = ggml_get_rows(ctx, model.speech_emb, model.builtin_cond_prompt_tokens);
    ggml_tensor * temb = ggml_get_rows(ctx, model.text_emb, text_tokens);
    ggml_tensor * semb = ggml_get_rows(ctx, model.speech_emb, start_token);

    ggml_tensor * inp = ggml_concat(ctx, spkr, cond, 1);
    inp = ggml_concat(ctx, inp, temb, 1);
    inp = ggml_concat(ctx, inp, semb, 1);
    inp = ggml_add(ctx, inp, ggml_get_rows(ctx, model.wpe, position));

    build_transformer_core(ctx, gf, model, inp, 0, N);
    ggml_free(ctx);
    return gf;
}

static ggml_cgraph * build_step_graph(const chatterbox_model & model, int n_past) {
    static size_t buf_size = ggml_tensor_overhead()*CHBX_MAX_NODES + ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * speech_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_token, "speech_token"); ggml_set_input(speech_token);
    ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(position, "position"); ggml_set_input(position);

    ggml_tensor * inp = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, speech_token),
        ggml_get_rows(ctx, model.wpe, position));

    build_transformer_core(ctx, gf, model, inp, n_past, 1);
    ggml_free(ctx);
    return gf;
}

// --------------------------------------------------------------------------
// Evaluation
// --------------------------------------------------------------------------

static bool eval_prompt(
    const chatterbox_model & model, ggml_gallocr_t allocr, int n_threads,
    const std::vector<int32_t> & text_tokens, std::vector<float> & logits_out, int & prompt_len) {

    prompt_len = 1 + model.hparams.cond_prompt_len + (int)text_tokens.size() + 1;
    if (prompt_len > model.hparams.n_ctx) {
        fprintf(stderr, "%s: prompt %d exceeds context %d\n", __func__, prompt_len, model.hparams.n_ctx);
        return false;
    }
    ggml_cgraph * gf = build_prompt_graph(model, (int)text_tokens.size());
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "text_tokens"), text_tokens.data(), 0, text_tokens.size()*sizeof(int32_t));
    int32_t st = model.hparams.start_speech_token;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &st, 0, sizeof(st));
    std::vector<int32_t> pos(prompt_len);
    for (int i = 0; i < prompt_len; ++i) pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), pos.data(), 0, pos.size()*sizeof(int32_t));

    if (ggml_backend_is_cpu(model.backend)) ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(model.hparams.n_speech_vocab);
    ggml_backend_tensor_get(logits, logits_out.data(),
        (size_t)model.hparams.n_speech_vocab*(prompt_len-1)*sizeof(float),
        (size_t)model.hparams.n_speech_vocab*sizeof(float));
    return true;
}

static bool eval_step(
    const chatterbox_model & model, ggml_gallocr_t allocr, int n_threads,
    int n_past, int32_t token, std::vector<float> & logits_out) {

    ggml_cgraph * gf = build_step_graph(model, n_past);
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &token, 0, sizeof(token));
    int32_t position = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), &position, 0, sizeof(position));

    if (ggml_backend_is_cpu(model.backend)) ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(model.hparams.n_speech_vocab);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, (size_t)model.hparams.n_speech_vocab*sizeof(float));
    return true;
}

// --------------------------------------------------------------------------
// Sampling
// --------------------------------------------------------------------------

// Matches HuggingFace LogitsProcessorList order used in inference_turbo:
//   1. TemperatureLogitsWarper   (if temp > 0 and temp != 1)
//   2. TopKLogitsWarper          (if top_k > 0)
//   3. TopPLogitsWarper          (if top_p < 1)
//   4. RepetitionPenaltyLogitsProcessor (if penalty != 1)
// Then softmax + multinomial.
static int32_t sample_next_token(
    const std::vector<float> & logits,
    const std::vector<int32_t> & generated,
    const cli_params & params,
    std::mt19937 & rng) {

    const int n = (int)logits.size();
    std::vector<float> scores(logits.begin(), logits.end());

    // 1. Temperature
    if (params.temp > 0.0f && params.temp != 1.0f) {
        float inv_t = 1.0f / params.temp;
        for (float & s : scores) s *= inv_t;
    }

    // 2. TopK  — set everything outside the top-k to -inf
    if (params.top_k > 0 && params.top_k < n) {
        std::vector<float> tmp(scores);
        std::nth_element(tmp.begin(), tmp.begin() + params.top_k, tmp.end(), std::greater<float>());
        float threshold = tmp[params.top_k];
        int kept = 0;
        for (float s : scores) if (s > threshold) ++kept;
        if (kept < params.top_k) threshold -= 1e-10f;
        for (float & s : scores) if (s <= threshold) s = -INFINITY;
    }

    // 3. TopP — set tokens below the cumulative probability cutoff to -inf
    if (params.top_p < 1.0f) {
        struct IS { int idx; float s; };
        std::vector<IS> sorted;
        sorted.reserve(n);
        for (int i = 0; i < n; ++i) if (scores[i] != -INFINITY) sorted.push_back({i, scores[i]});
        std::sort(sorted.begin(), sorted.end(), [](const IS& a, const IS& b){ return a.s > b.s; });

        float mx = sorted.empty() ? 0.0f : sorted[0].s;
        std::vector<float> probs(sorted.size());
        float psum = 0;
        for (size_t i = 0; i < sorted.size(); ++i) { probs[i] = std::exp(sorted[i].s - mx); psum += probs[i]; }
        for (float & p : probs) p /= psum;

        float cum = 0;
        std::set<int> keep_set;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += probs[i];
            keep_set.insert(sorted[i].idx);
            if (cum >= params.top_p) break;
        }
        if (keep_set.empty() && !sorted.empty()) keep_set.insert(sorted[0].idx);
        for (int i = 0; i < n; ++i) if (keep_set.find(i) == keep_set.end()) scores[i] = -INFINITY;
    }

    // 4. Repetition penalty (HF convention: divide positive, multiply negative)
    if (params.repeat_penalty != 1.0f && !generated.empty()) {
        std::set<int32_t> seen(generated.begin(), generated.end());
        for (int32_t t : seen) {
            if (t < 0 || t >= n) continue;
            if (scores[t] == -INFINITY) continue;
            scores[t] = scores[t] > 0 ? scores[t] / params.repeat_penalty : scores[t] * params.repeat_penalty;
        }
    }

    // Softmax + sample (or argmax for greedy)
    float mx = -INFINITY;
    for (float s : scores) if (s != -INFINITY) mx = std::max(mx, s);

    std::vector<float> probs(n);
    float psum = 0;
    for (int i = 0; i < n; ++i) {
        probs[i] = (scores[i] == -INFINITY) ? 0.0f : std::exp(scores[i] - mx);
        psum += probs[i];
    }
    if (psum == 0.0f) return 0;
    for (float & p : probs) p /= psum;

    if (params.temp <= 0.0f) {
        return (int32_t)std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

int main(int argc, char ** argv) {
    ggml_time_init();
    cli_params params;
    if (!parse_args(argc, argv, params)) { print_usage(argv[0]); return 1; }

    try {
        // Short-circuit: user gave us speech tokens directly + --s3gen-gguf. Skip T3 entirely.
        if (params.model.empty() && !params.s3gen_gguf.empty() && !params.tokens_file.empty()) {
            std::vector<int32_t> speech_tokens = read_token_file(params.tokens_file);
            if (speech_tokens.empty()) throw std::runtime_error("empty speech tokens file");
            s3gen_synthesize_opts opts;
            opts.s3gen_gguf_path = params.s3gen_gguf;
            opts.out_wav_path    = params.out_wav;
            opts.ref_dir         = params.ref_dir;
            opts.seed            = params.seed;
            opts.n_threads       = params.n_threads;
            opts.debug           = params.debug;
            if (!params.reference_audio.empty()) {
                if (!compute_prompt_feat_native(params.reference_audio, params.s3gen_gguf,
                                                opts.prompt_feat_override,
                                                opts.prompt_feat_rows_override))
                    throw std::runtime_error("failed to compute prompt_feat from --reference-audio");
                // Best-effort: try to compute the S3Gen `embedding` natively too.
                // Falls through to ref_dir/embedding.npy if the s3gen GGUF is pre-A1-2d-a.
                (void)compute_embedding_native(params.reference_audio, params.s3gen_gguf,
                                               opts.embedding_override);
                // And the S3Gen-side prompt_token via S3TokenizerV2 (Phase 2e).
                std::vector<int32_t> dummy_cond;
                (void)compute_speech_tokens_native(params.reference_audio, params.s3gen_gguf,
                                                   /*max_cond_tokens=*/-1,
                                                   opts.prompt_token_override, dummy_cond,
                                                   params.n_threads);
            }
            return s3gen_synthesize_to_wav(speech_tokens, opts);
        }

        // Load model first so we can use the GGUF-embedded tokenizer (if any).
        chatterbox_model model;
        const int64_t _t3_load_t0 = ggml_time_us();
        if (!load_model_gguf(params.model, model, params.n_ctx, params.n_gpu_layers)) return 1;
        const int64_t _t3_load_ms = (ggml_time_us() - _t3_load_t0) / 1000;
        fprintf(stderr, "BENCH: T3_LOAD_MS=%lld\n", (long long)_t3_load_ms);

        // Voice-profile override on the T3 side.  We resolve two tensors
        // independently:
        //
        //   speaker_emb           — take from ref_dir/speaker_emb.npy if
        //                           available, otherwise compute in C++ from
        //                           --reference-audio via VoiceEncoder.
        //   cond_prompt_tokens    — only available from ref_dir (until the
        //                           S3TokenizerV2 C++ port in Phase 2e).
        //
        // The S3Gen side is overridden later inside s3gen_synthesize_to_wav.
        bool have_se = false, have_ct = false;
        std::vector<float>   se_data;
        std::vector<int32_t> ct_data;
        if (!params.ref_dir.empty()) {
            const std::string se_path = params.ref_dir + "/speaker_emb.npy";
            const std::string ct_path = params.ref_dir + "/cond_prompt_speech_tokens.npy";
            if (file_exists(se_path)) {
                npy_array se = npy_load(se_path);
                se_data.assign((const float *)se.data.data(),
                               (const float *)se.data.data() + se.n_elements());
                have_se = true;
            }
            if (file_exists(ct_path)) {
                npy_array ct = npy_load(ct_path);
                ct_data.assign((const int32_t *)ct.data.data(),
                               (const int32_t *)ct.data.data() + ct.n_elements());
                have_ct = true;
            }
        }
        if (!have_se && !params.reference_audio.empty()) {
            voice_encoder_weights vew;
            if (voice_encoder_load(params.model, vew)) {
                fprintf(stderr, "voice_encoder: computing speaker_emb from %s\n",
                        params.reference_audio.c_str());
                std::vector<float> wav;
                int sr = 0;
                if (!wav_load(params.reference_audio, wav, sr))
                    throw std::runtime_error("failed to load --reference-audio for VoiceEncoder");
                if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
                if (!voice_encoder_embed(wav, vew, se_data))
                    throw std::runtime_error("VoiceEncoder forward failed");
                have_se = true;
            } else {
                fprintf(stderr,
                    "voice_encoder: T3 GGUF has no VE weights; cannot synthesise speaker_emb natively "
                    "(re-run scripts/convert-t3-turbo-to-gguf.py)\n");
            }
        }

        // Speech-token overrides: compute both cond_prompt_speech_tokens
        // (T3 side) and prompt_token (S3Gen side, stashed for later) via
        // the C++ S3TokenizerV2 port if --reference-audio is given and the
        // s3gen GGUF has the tokenizer weights (Phase 2e).
        std::vector<int32_t> prompt_token_from_ref;
        bool ct_from_cpp = false;
        if (!have_ct && !params.reference_audio.empty() && !params.s3gen_gguf.empty()) {
            std::vector<int32_t> cond_tokens;
            if (compute_speech_tokens_native(params.reference_audio, params.s3gen_gguf,
                                             /*max_cond_tokens=*/model.hparams.cond_prompt_len,
                                             prompt_token_from_ref, cond_tokens,
                                             params.n_threads)) {
                ct_data = std::move(cond_tokens);
                have_ct = true;
                ct_from_cpp = true;
            }
        }

        if (have_se) {
            if ((int64_t)se_data.size() != ggml_nelements(model.builtin_speaker_emb)) {
                fprintf(stderr,
                    "error: speaker_emb has %zu elements but builtin_speaker_emb expects %lld\n",
                    se_data.size(), (long long)ggml_nelements(model.builtin_speaker_emb));
                return 1;
            }
            ggml_backend_tensor_set(model.builtin_speaker_emb,
                                    se_data.data(), 0, ggml_nbytes(model.builtin_speaker_emb));
        }

        if (have_ct) {
            if ((int64_t)ct_data.size() == ggml_nelements(model.builtin_cond_prompt_tokens)) {
                ggml_backend_tensor_set(model.builtin_cond_prompt_tokens,
                                        ct_data.data(), 0,
                                        ggml_nbytes(model.builtin_cond_prompt_tokens));
            } else {
                ggml_init_params op = { ggml_tensor_overhead() * 2, nullptr, true };
                model.ctx_override = ggml_init(op);
                if (!model.ctx_override) throw std::runtime_error("ggml_init(ctx_override) failed");
                ggml_tensor * new_ct = ggml_new_tensor_1d(model.ctx_override, GGML_TYPE_I32, (int64_t)ct_data.size());
                ggml_set_name(new_ct, "chatterbox/builtin/cond_prompt_speech_tokens_override");
                model.buffer_override = ggml_backend_alloc_ctx_tensors(model.ctx_override, model.backend);
                if (!model.buffer_override) throw std::runtime_error("alloc override buffer failed");
                ggml_backend_tensor_set(new_ct, ct_data.data(), 0, ct_data.size() * sizeof(int32_t));
                model.builtin_cond_prompt_tokens = new_ct;
                model.hparams.cond_prompt_len = (int32_t)ct_data.size();
            }
        }

        if (have_se || have_ct) {
            fprintf(stderr,
                "%s: T3 voice override — speaker_emb=%s, cond_prompt_tokens=%s\n",
                __func__,
                have_se ? (params.reference_audio.empty() ? "ref_dir" : "C++ VoiceEncoder") : "built-in",
                have_ct ? (ct_from_cpp ? "C++ S3TokenizerV2" : "ref_dir") : "built-in");
        } else if (!params.ref_dir.empty() || !params.reference_audio.empty()) {
            fprintf(stderr,
                "%s: no T3 override; keeping built-in T3 voice\n", __func__);
        }

        std::vector<int32_t> text_tokens;
        if (!params.text.empty()) {
            if (model.tok_tokens.empty()) {
                fprintf(stderr,
                    "error: this GGUF has no embedded tokenizer. Re-run\n"
                    "       scripts/convert-t3-turbo-to-gguf.py to produce a fresh GGUF\n"
                    "       with tokenizer.ggml.* metadata.\n");
                return 1;
            }
            gpt2_bpe bpe;
            bpe.load_from_arrays(model.tok_tokens, model.tok_merges);

            std::string normalized = gpt2_bpe::punc_norm(params.text);
            text_tokens = bpe.tokenize(normalized);

            if (params.dump_tokens_only) {
                for (size_t i = 0; i < text_tokens.size(); ++i) {
                    if (i) printf(",");
                    printf("%d", text_tokens[i]);
                }
                printf("\n");
                return 0;
            }

            fprintf(stderr, "%s: text: \"%s\"\n", __func__, normalized.c_str());
            fprintf(stderr, "%s: %zu text tokens\n", __func__, text_tokens.size());
        } else {
            text_tokens = read_token_file(params.tokens_file);
        }
        if (text_tokens.empty()) throw std::runtime_error("empty token input");

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        std::mt19937 rng(params.seed);
        std::vector<float> logits;
        int prompt_len = 0;
        const int64_t _t3_infer_t0 = ggml_time_us();
        if (!eval_prompt(model, allocr, params.n_threads, text_tokens, logits, prompt_len))
            throw std::runtime_error("prompt eval failed");

        int n_past = prompt_len;
        std::vector<int32_t> generated;
        generated.reserve(params.n_predict + 1);

        int32_t current = sample_next_token(logits, generated, params, rng);
        generated.push_back(current);

        for (int i = 0; i < params.n_predict; ++i) {
            if (current == model.hparams.stop_speech_token) break;
            if (n_past + 1 > model.hparams.n_ctx) { fprintf(stderr, "KV cache full\n"); break; }
            if (!eval_step(model, allocr, params.n_threads, n_past, current, logits))
                throw std::runtime_error("step eval failed");
            ++n_past;
            current = sample_next_token(logits, generated, params, rng);
            generated.push_back(current);
        }

        if (!generated.empty() && generated.back() == model.hparams.stop_speech_token)
            generated.pop_back();

        const int64_t _t3_infer_ms = (ggml_time_us() - _t3_infer_t0) / 1000;
        fprintf(stderr, "BENCH: T3_INFER_MS=%lld tokens=%zu\n",
                (long long)_t3_infer_ms, generated.size());

        if (!params.output.empty()) write_token_file(params.output, generated);

        // If --s3gen-gguf is set, chain into the S3Gen + HiFT vocoder to write a wav.
        if (!params.s3gen_gguf.empty()) {
            s3gen_synthesize_opts opts;
            opts.s3gen_gguf_path = params.s3gen_gguf;
            opts.out_wav_path    = params.out_wav;
            opts.ref_dir         = params.ref_dir;
            opts.seed            = params.seed;
            opts.n_threads       = params.n_threads;
            opts.debug           = params.debug;
            if (!params.reference_audio.empty()) {
                if (!compute_prompt_feat_native(params.reference_audio, params.s3gen_gguf,
                                                opts.prompt_feat_override,
                                                opts.prompt_feat_rows_override))
                    throw std::runtime_error("failed to compute prompt_feat from --reference-audio");
                (void)compute_embedding_native(params.reference_audio, params.s3gen_gguf,
                                               opts.embedding_override);
                if (!prompt_token_from_ref.empty()) {
                    opts.prompt_token_override = std::move(prompt_token_from_ref);
                }
            }
            int rc = s3gen_synthesize_to_wav(generated, opts);
            if (rc != 0) return rc;
        } else {
            // Legacy: print the tokens to stdout too (handy for piping).
            for (size_t i = 0; i < generated.size(); ++i) { if (i) printf(","); printf("%d", generated[i]); }
            printf("\n");
        }

        ggml_gallocr_free(allocr);
        ggml_backend_buffer_free(model.buffer_w);
        ggml_backend_buffer_free(model.buffer_kv);
        if (model.buffer_override) ggml_backend_buffer_free(model.buffer_override);
        ggml_backend_free(model.backend);
        ggml_free(model.ctx_w);
        ggml_free(model.ctx_kv);
        if (model.ctx_override) ggml_free(model.ctx_override);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
