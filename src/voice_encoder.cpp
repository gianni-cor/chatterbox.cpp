#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ============================================================================
// GGUF loader
// ============================================================================

static bool copy_tensor_f32(ggml_context * ctx, const char * name,
                            std::vector<float> & out)
{
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) return false;
    out.resize(ggml_nelements(t));
    std::memcpy(out.data(), ggml_get_data(t), ggml_nbytes(t));
    return true;
}

bool voice_encoder_load(const std::string & t3_gguf_path,
                        voice_encoder_weights & out)
{
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(t3_gguf_path.c_str(), gp);
    if (!g) {
        fprintf(stderr, "voice_encoder_load: failed to open %s\n", t3_gguf_path.c_str());
        return false;
    }

    // Presence check: the VE weights landed in Phase 2c of the A1 plan, so a
    // pre-A1 GGUF won't have them.  Bail cleanly.
    if (gguf_find_key(g, "voice_encoder.hidden_size") < 0) {
        gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }

    auto get_u32 = [&](const char * k, uint32_t fallback) -> uint32_t {
        int64_t id = gguf_find_key(g, k);
        return id < 0 ? fallback : gguf_get_val_u32(g, id);
    };
    auto get_f32 = [&](const char * k, float fallback) -> float {
        int64_t id = gguf_find_key(g, k);
        return id < 0 ? fallback : gguf_get_val_f32(g, id);
    };

    out.n_layers       = (int)get_u32("voice_encoder.num_layers",    3);
    out.n_mels         = (int)get_u32("voice_encoder.n_mels",       40);
    out.hidden         = (int)get_u32("voice_encoder.hidden_size",  256);
    out.embedding      = (int)get_u32("voice_encoder.embedding_size", out.hidden);
    out.partial_frames = (int)get_u32("voice_encoder.partial_frames", 160);
    out.overlap        = get_f32("voice_encoder.overlap",            0.5f);
    out.rate           = get_f32("voice_encoder.rate",               1.3f);
    out.min_coverage   = get_f32("voice_encoder.min_coverage",       0.8f);

    out.lstm.clear();
    out.lstm.resize(out.n_layers);
    for (int l = 0; l < out.n_layers; ++l) {
        auto & L = out.lstm[l];
        L.H = out.hidden;
        L.I = (l == 0) ? out.n_mels : out.hidden;
        char name[128];
        std::snprintf(name, sizeof(name), "voice_encoder/lstm/weight_ih_l%d", l);
        if (!copy_tensor_f32(tmp_ctx, name, L.w_ih)) goto fail;
        std::snprintf(name, sizeof(name), "voice_encoder/lstm/weight_hh_l%d", l);
        if (!copy_tensor_f32(tmp_ctx, name, L.w_hh)) goto fail;
        std::snprintf(name, sizeof(name), "voice_encoder/lstm/bias_ih_l%d", l);
        if (!copy_tensor_f32(tmp_ctx, name, L.b_ih)) goto fail;
        std::snprintf(name, sizeof(name), "voice_encoder/lstm/bias_hh_l%d", l);
        if (!copy_tensor_f32(tmp_ctx, name, L.b_hh)) goto fail;
    }
    if (!copy_tensor_f32(tmp_ctx, "voice_encoder/proj/weight", out.proj_w)) goto fail;
    if (!copy_tensor_f32(tmp_ctx, "voice_encoder/proj/bias",   out.proj_b)) goto fail;
    if (!copy_tensor_f32(tmp_ctx, "voice_encoder/mel_fb",      out.mel_fb)) goto fail;

    gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
    return true;

fail:
    fprintf(stderr, "voice_encoder_load: missing expected tensor in %s\n", t3_gguf_path.c_str());
    gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
    return false;
}

// ============================================================================
// LSTM forward (single layer, unidirectional)
// ============================================================================

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Compute h_T (final hidden state) after running a single-layer unidirectional
// LSTM over an input sequence of length T.
//
//   x          : row-major (T, I)
//   h_seq      : row-major (T, H), output per-step hidden states (or nullptr)
//   h_last     : (H,) final hidden state after T steps (always filled)
//   c_last     : (H,) final cell state (always filled)
//   scratch    : (4H,) workspace for gate pre-activations
//
// PyTorch's LSTMCell computes
//   gates = x @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh
// with gates rows ordered [i, f, g, o] (see torch.nn.LSTM docs).
static void lstm_layer_forward(
    const voice_encoder_lstm_layer & L,
    const float * x, int T,
    float * h_seq,   // (T, H) or nullptr
    float * h_last,  // (H,)
    float * c_last   // (H,)
) {
    const int H = L.H;
    const int I = L.I;
    const int G = 4 * H;

    std::vector<float> gates(G);
    std::vector<float> h_prev(H, 0.0f);
    std::vector<float> c_prev(H, 0.0f);

    for (int t = 0; t < T; ++t) {
        const float * x_t = x + (size_t)t * I;

        // gates = b_ih + b_hh + W_ih @ x_t + W_hh @ h_prev.
        //   W_ih is stored as (G, I) row-major → row g starts at g*I.
        //   Same for W_hh at (G, H).
        for (int g = 0; g < G; ++g) gates[g] = L.b_ih[g] + L.b_hh[g];

        for (int g = 0; g < G; ++g) {
            const float * row = L.w_ih.data() + (size_t)g * I;
            float acc = 0.0f;
            for (int i = 0; i < I; ++i) acc += row[i] * x_t[i];
            gates[g] += acc;
        }
        for (int g = 0; g < G; ++g) {
            const float * row = L.w_hh.data() + (size_t)g * H;
            float acc = 0.0f;
            for (int i = 0; i < H; ++i) acc += row[i] * h_prev[i];
            gates[g] += acc;
        }

        // Split gates into [i, f, g, o] chunks of H each; apply activations.
        //   c_t = f * c_prev + i * g_raw
        //   h_t = o * tanh(c_t)
        for (int h = 0; h < H; ++h) {
            float i_t = sigmoidf(gates[0*H + h]);
            float f_t = sigmoidf(gates[1*H + h]);
            float g_t = std::tanh(gates[2*H + h]);
            float o_t = sigmoidf(gates[3*H + h]);

            float c_t = f_t * c_prev[h] + i_t * g_t;
            float h_t = o_t * std::tanh(c_t);

            c_prev[h] = c_t;
            h_prev[h] = h_t;
        }

        if (h_seq) std::memcpy(h_seq + (size_t)t * H, h_prev.data(), H * sizeof(float));
    }

    std::memcpy(h_last, h_prev.data(), H * sizeof(float));
    std::memcpy(c_last, c_prev.data(), H * sizeof(float));
}

// ============================================================================
// VoiceEncoder forward
// ============================================================================

// Project + ReLU + L2-norm on a single 256-d hidden-state vector.
static void project_and_normalise(
    const voice_encoder_weights & w,
    const float * h_in,   // (hidden,)
    float * out           // (embedding,)
) {
    const int H = w.hidden;
    const int E = w.embedding;
    for (int o = 0; o < E; ++o) {
        const float * row = w.proj_w.data() + (size_t)o * H;
        float acc = w.proj_b[o];
        for (int h = 0; h < H; ++h) acc += row[h] * h_in[h];
        out[o] = acc;
    }
    // ReLU (ve_final_relu = True).
    for (int o = 0; o < E; ++o) if (out[o] < 0.0f) out[o] = 0.0f;
    // L2-normalise.
    double sq = 0.0;
    for (int o = 0; o < E; ++o) sq += (double)out[o] * (double)out[o];
    double n = std::sqrt(sq);
    if (n > 1e-12) {
        float s = (float)(1.0 / n);
        for (int o = 0; o < E; ++o) out[o] *= s;
    }
}

// Pick partial-window step size / count to match VoiceEncoder.inference +
// get_num_wins exactly.
static void compute_partials(int n_frames, int partial, float rate,
                             int sample_rate_hz,
                             float overlap, float min_coverage,
                             int & n_wins, int & step, int & target_n)
{
    // voice_encoder/get_frame_step:
    //   rate != None → frame_step = int(round((sr / rate) / partial))
    //   rate == None → frame_step = int(round(partial * (1 - overlap)))
    if (rate > 0.0f) {
        step = (int)std::lround(((double)sample_rate_hz / (double)rate) / (double)partial);
    } else {
        step = (int)std::lround((double)partial * (1.0 - overlap));
    }
    if (step <= 0) step = 1;
    if (step > partial) step = partial;

    // get_num_wins:
    //   n_wins, remainder = divmod(max(n_frames - partial + step, 0), step)
    //   if n_wins == 0 or (remainder + (partial - step)) / partial >= min_coverage:
    //       n_wins += 1
    int a = std::max(n_frames - partial + step, 0);
    int nw = a / step;
    int remainder = a - nw * step;
    if (nw == 0 || ((double)(remainder + (partial - step)) / (double)partial) >= (double)min_coverage) {
        nw += 1;
    }
    n_wins   = nw;
    target_n = partial + step * (nw - 1);
}

bool voice_encoder_embed(const std::vector<float> & wav_16k,
                         const voice_encoder_weights & w,
                         std::vector<float> & out)
{
    if (w.mel_fb.empty() || w.lstm.size() != (size_t)w.n_layers) {
        fprintf(stderr, "voice_encoder_embed: weights are incomplete\n");
        return false;
    }

    // 1. Compute the VE mel (40-ch power spec at 16 kHz, center=True) →
    //    shape (T, 40) row-major.  T = 1 + L/160.
    std::vector<float> mel = mel_extract_16k_40(wav_16k, w.mel_fb);
    if (mel.empty()) {
        fprintf(stderr, "voice_encoder_embed: mel extraction failed\n");
        return false;
    }
    const int T_mel = (int)(mel.size() / w.n_mels);

    // 2. Compute partial-window layout.
    int n_wins, step, target_n;
    compute_partials(T_mel, w.partial_frames, w.rate, w.partial_frames, w.overlap, w.min_coverage,
                     n_wins, step, target_n);
    // Pad mel up to target_n if needed (with zeros, same as VoiceEncoder.inference).
    if (target_n > T_mel) {
        mel.resize((size_t)target_n * w.n_mels, 0.0f);
    } else if (target_n < T_mel) {
        // Trim.
        mel.resize((size_t)target_n * w.n_mels);
    }
    const int T_total = target_n;
    (void)T_total;

    // 3. For each partial: run the 3-layer unidirectional LSTM over 160 frames,
    //    take the last layer's final hidden state, project + ReLU + L2-norm.
    //    Accumulate the 256-d vectors, then mean + L2-norm → speaker embedding.
    const int H = w.hidden;
    const int E = w.embedding;
    const int partial = w.partial_frames;

    std::vector<float> h_prev_layer_seq(partial * H);   // layer 1 / 2 input
    std::vector<float> h_cur_layer_seq (partial * H);
    std::vector<float> h_last(H);
    std::vector<float> c_last(H);

    std::vector<float> emb_accum(E, 0.0f);

    for (int wi = 0; wi < n_wins; ++wi) {
        const int t0 = wi * step;
        const float * window_mel = mel.data() + (size_t)t0 * w.n_mels;

        // Layer 0: input is the mel (I = n_mels).
        lstm_layer_forward(w.lstm[0],
                           window_mel, partial,
                           w.n_layers > 1 ? h_cur_layer_seq.data() : nullptr,
                           h_last.data(), c_last.data());

        // Intermediate layers: input is h_seq from previous layer.
        for (int l = 1; l < w.n_layers; ++l) {
            std::swap(h_prev_layer_seq, h_cur_layer_seq);
            const bool last_layer = (l == w.n_layers - 1);
            lstm_layer_forward(w.lstm[l],
                               h_prev_layer_seq.data(), partial,
                               last_layer ? nullptr : h_cur_layer_seq.data(),
                               h_last.data(), c_last.data());
        }

        std::vector<float> emb(E);
        project_and_normalise(w, h_last.data(), emb.data());
        for (int o = 0; o < E; ++o) emb_accum[o] += emb[o];
    }

    // Mean over partials, L2-norm.  Matches VoiceEncoder.inference's
    // torch.mean(partial_embeds[start:end], dim=0) followed by the final
    // L2-normalisation.
    float inv_n = 1.0f / (float)n_wins;
    for (int o = 0; o < E; ++o) emb_accum[o] *= inv_n;

    double sq = 0.0;
    for (int o = 0; o < E; ++o) sq += (double)emb_accum[o] * (double)emb_accum[o];
    double n = std::sqrt(sq);
    if (n > 1e-12) {
        float s = (float)(1.0 / n);
        for (int o = 0; o < E; ++o) emb_accum[o] *= s;
    }

    out = std::move(emb_accum);
    return true;
}
