// Chatterbox multilingual T3 (Llama-520M) variant: loader + forward pass.
//
// Structural differences from the GPT-2 Medium Turbo variant in src/main.cpp:
//   - 30 layers vs 24, head_dim=64, n_kv_head=16 (MHA, not GQA).
//   - Pre-norm with RMSNorm (no bias) instead of LayerNorm.
//   - Rotary position embedding with llama3 scaling: freq_factors precomputed
//     at load time and applied through ggml_rope_ext's `c` param.
//   - SwiGLU MLP: SiLU(gate(x)) * up(x) -> down(x); three Linears per layer.
//   - Separate Q/K/V projections (no fused c_attn).
//   - Classifier-Free Guidance: each T3 graph runs twice per call, once for
//     the conditional (full text embeddings) batch element and once for the
//     unconditional one (text embeddings zeroed).  Two independent KV caches
//     live inside the model struct for this.  Logits are combined in the
//     sampler as `cond + cfg_weight * (cond - uncond)`.
//   - Conditioning tokens:
//        spkr_enc(speaker_emb)             -> 1 token
//        perceiver(cond_prompt_speech_emb) -> 32 tokens (shared AttentionBlock2
//                                             used cross-attn then self-attn)
//        emotion_adv_fc(exaggeration)      -> 1 token
//     These concatenate into `cond_emb` (34 tokens).  Conditional and
//     unconditional passes share the cond_emb; text/speech embeddings differ
//     between them (uncond zeroes text embeds but keeps the speech BOS).

#include "chatterbox_t3_internal.h"
#include "t3_mtl.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::chatterbox::detail {

namespace {

int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}

ggml_tensor * require_tensor(const chatterbox_model & m, const char * name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end() || !it->second) {
        throw std::runtime_error(std::string("missing tensor: ") + name);
    }
    return it->second;
}

uint32_t get_u32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_u32(ctx, require_key(ctx, key));
}
float get_f32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_f32(ctx, require_key(ctx, key));
}
bool get_bool(const gguf_context * ctx, const char * key) {
    return gguf_get_val_bool(ctx, require_key(ctx, key));
}

// Llama-3 style RoPE frequency scaling (transformers `_compute_llama3_parameters`).
// Produces a per-frequency-bin correction factor that ggml_rope_ext will
// apply as its `c` (freq_factors) parameter.  Length is head_dim/2.
//
//   base_inv_freq[i] = 1 / theta^(2i / head_dim)
//   wavelen[i]       = 2*pi / base_inv_freq[i]
//   if wavelen > low_wavelen:  inv_freq = base / factor
//   if wavelen < high_wavelen: inv_freq = base
//   else:                      smooth transition
//   freq_factor[i]   = base_inv_freq[i] / effective_inv_freq[i]
//                    (ggml divides each band's base frequency by this).
std::vector<float> compute_llama3_freq_factors(int head_dim, float theta,
                                               float factor, float low_freq,
                                               float high_freq, int orig_max_pos) {
    const int half = head_dim / 2;
    std::vector<float> ff(half, 1.0f);

    const float low_wavelen  = (float) orig_max_pos / low_freq;
    const float high_wavelen = (float) orig_max_pos / high_freq;

    for (int i = 0; i < half; ++i) {
        const float base = 1.0f / std::pow(theta, (float)(2 * i) / (float) head_dim);
        const float wavelen = 2.0f * (float) M_PI / base;

        float effective;
        if (wavelen > low_wavelen) {
            effective = base / factor;
        } else if (wavelen < high_wavelen) {
            effective = base;
        } else {
            const float smooth = ((float) orig_max_pos / wavelen - low_freq) /
                                 (high_freq - low_freq);
            const float scaled = base / factor;
            effective = (1.0f - smooth) * scaled + smooth * base;
        }
        ff[i] = base / effective;
    }
    return ff;
}

// Perceiver cross/self attention block (Perceiver.attn): a single
// AttentionBlock2 with LayerNorm + 4-head scaled-dot-product attention +
// proj_out + residual to the query side.
//
// In the perceiver forward we call this twice with the same weights:
//   pre_att = attn(query_tokens, h_in)       // cross-attn
//   out     = attn(pre_att, pre_att)         // self-attn
//
// perc_q shape:  (n_embd, T_q)      query input (added to the output as residual)
// perc_kv shape: (n_embd, T_kv)     key/value input
// Returns:       (n_embd, T_q)
ggml_tensor * build_perceiver_attn(ggml_context * ctx,
                                   const perceiver_weights & w,
                                   const chatterbox_hparams & hp,
                                   ggml_tensor * perc_q,
                                   ggml_tensor * perc_kv) {
    const int n_embd  = hp.n_embd;
    const int n_heads = hp.perceiver_heads;
    const int head_dim = n_embd / n_heads;

    const int T_q  = perc_q->ne[1];
    const int T_kv = perc_kv->ne[1];

    // LayerNorm on both inputs (same affine weights as Python's self.norm).
    auto ln = [&](ggml_tensor * x) {
        ggml_tensor * n = ggml_norm(ctx, x, /*eps=*/1e-5f);
        return ggml_add(ctx, ggml_mul(ctx, n, w.norm_g), w.norm_b);
    };
    ggml_tensor * q_norm = ln(perc_q);
    ggml_tensor * kv_norm = ln(perc_kv);

    ggml_tensor * q_lin = ggml_add(ctx, ggml_mul_mat(ctx, w.to_q_w, q_norm),  w.to_q_b);
    ggml_tensor * k_lin = ggml_add(ctx, ggml_mul_mat(ctx, w.to_k_w, kv_norm), w.to_k_b);
    ggml_tensor * v_lin = ggml_add(ctx, ggml_mul_mat(ctx, w.to_v_w, kv_norm), w.to_v_b);

    // Reshape to (head_dim, T, n_heads) for flash_attn_ext.
    ggml_tensor * Q = ggml_reshape_3d(ctx, q_lin, head_dim, n_heads, T_q);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // (head_dim, T_q, n_heads)
    ggml_tensor * K = ggml_reshape_3d(ctx, k_lin, head_dim, n_heads, T_kv);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    ggml_tensor * V = ggml_reshape_3d(ctx, v_lin, head_dim, n_heads, T_kv);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    const float scale = 1.0f / std::sqrt((float) head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, /*mask=*/nullptr,
                                             scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    // attn output layout: (head_dim, n_heads, T_q, 1)
    attn = ggml_reshape_2d(ctx, attn, n_embd, T_q);

    ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, w.proj_out_w, attn), w.proj_out_b);
    return ggml_add(ctx, perc_q, proj);
}

// Perceiver forward: pre_att = attn(pre_attention_query, h); return attn(pre_att, pre_att)
// pre_attention_query shape in the GGUF: (1024, 32, 1) after transpose convention
// h shape (cond_prompt_speech_emb): (1024, cond_prompt_len)
ggml_tensor * build_perceiver(ggml_context * ctx,
                              const chatterbox_model & m,
                              ggml_tensor * h) {
    // pre_attention_query stored as (1, 32, 1024) → ggml storage (1024, 32, 1).
    // Take it as a (1024, 32) 2D tensor.
    ggml_tensor * query = ggml_reshape_2d(ctx, m.perceiver.pre_attention_query, m.hparams.n_embd, m.hparams.perceiver_queries);
    ggml_tensor * pre_att = build_perceiver_attn(ctx, m.perceiver, m.hparams, query, h);
    ggml_tensor * out     = build_perceiver_attn(ctx, m.perceiver, m.hparams, pre_att, pre_att);
    return out;
}

// One Llama transformer block.  Writes K/V into the selected KV cache
// tensors at positions [n_past, n_past + N).
//
// inpL:       (n_embd, N)
// memory_k/v: 1D F32 buffers of size (head_dim * n_kv_head * n_ctx * n_layer)
ggml_tensor * build_llama_block(ggml_context * ctx, ggml_cgraph * gf,
                                const chatterbox_model & m,
                                int il,
                                ggml_tensor * inpL,
                                int n_past, int N,
                                ggml_tensor * memory_k,
                                ggml_tensor * memory_v,
                                ggml_tensor * pos_ids,
                                ggml_tensor * kq_mask) {
    const auto & hp = m.hparams;
    const auto & l  = m.layers_mtl[il];
    const int HD  = hp.head_dim;
    const int NH  = hp.n_head;
    const int NKV = hp.n_kv_head;
    const int n_ctx = hp.n_ctx;
    const int64_t L = n_past + N;

    const size_t kv_head_stride  = (size_t) HD * n_ctx * sizeof(float);
    const size_t kv_pos_stride   = (size_t) HD * sizeof(float);
    const size_t kv_layer_elems  = (size_t) HD * n_ctx * NKV;
    const size_t layer_off = (size_t) il * kv_layer_elems * sizeof(float);

    // Pre-attention RMSNorm (no bias).
    ggml_tensor * cur = ggml_rms_norm(ctx, inpL, hp.eps);
    cur = ggml_mul(ctx, cur, l.ln_attn_g);

    ggml_tensor * Qlin = ggml_mul_mat(ctx, l.wq, cur);  // (n_embd, N)
    ggml_tensor * Klin = ggml_mul_mat(ctx, l.wk, cur);
    ggml_tensor * Vlin = ggml_mul_mat(ctx, l.wv, cur);

    // Reshape to (HD, n_head, N). ggml_rope_ext requires ne[2] == len(pos_ids),
    // so sequence must be on ne[2] at the rope call.
    ggml_tensor * Q = ggml_reshape_3d(ctx, Qlin, HD, NH,  N);  // (HD, NH,  N)
    ggml_tensor * K = ggml_reshape_3d(ctx, Klin, HD, NKV, N);  // (HD, NKV, N)
    ggml_tensor * V = ggml_reshape_3d(ctx, Vlin, HD, NKV, N);  // (HD, NKV, N)

    // RoPE on Q and K (NEOX-style half-split convention used by Llama).
    const int rope_mode = GGML_ROPE_TYPE_NEOX;
    Q = ggml_rope_ext(ctx, Q, pos_ids, m.rope_freq_factors,
                      HD, rope_mode, hp.rope_orig_max_pos,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    K = ggml_rope_ext(ctx, K, pos_ids, m.rope_freq_factors,
                      HD, rope_mode, hp.rope_orig_max_pos,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // Flash attention (Turbo-style) expects (HD, N, NH).  Permute from
    // (HD, NH, N) -> (HD, N, NH) and then the KV cache keeps the same
    // [HD, n_ctx, n_head] layout used in src/main.cpp, so flash_attn can
    // read a contiguous slice without another permute at read time.
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // (HD, N,  NH)
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // (HD, N,  NKV)
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));  // (HD, N,  NKV)

    // Write K/V into the cache at [n_past : n_past+N) for this layer.
    {
        ggml_tensor * k_dst = ggml_view_3d(ctx, memory_k,
            HD, N, NKV,
            kv_pos_stride, kv_head_stride,
            layer_off + (size_t) n_past * kv_pos_stride);
        ggml_tensor * v_dst = ggml_view_3d(ctx, memory_v,
            HD, N, NKV,
            kv_pos_stride, kv_head_stride,
            layer_off + (size_t) n_past * kv_pos_stride);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
    }

    // Attention: read the full [0, L) slice from the cache.
    ggml_tensor * Kfull = ggml_view_3d(ctx, memory_k,
        HD, L, NKV,
        kv_pos_stride, kv_head_stride,
        layer_off);
    ggml_tensor * Vfull = ggml_view_3d(ctx, memory_v,
        HD, L, NKV,
        kv_pos_stride, kv_head_stride,
        layer_off);

    const float scale = 1.0f / std::sqrt((float) HD);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, Kfull, Vfull, kq_mask,
                                             scale, 0.0f, 0.0f);
    // attn: (HD, NH, N, 1) -> (n_embd, N)
    cur = ggml_reshape_2d(ctx, attn, hp.n_embd, N);

    // O-proj + residual.
    cur = ggml_mul_mat(ctx, l.wo, cur);
    cur = ggml_add(ctx, cur, inpL);

    // MLP (SwiGLU) with pre-norm + residual.
    ggml_tensor * inpFF = cur;
    ggml_tensor * norm2 = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), l.ln_mlp_g);
    ggml_tensor * gate  = ggml_silu(ctx, ggml_mul_mat(ctx, l.mlp_gate, norm2));
    ggml_tensor * up    = ggml_mul_mat(ctx, l.mlp_up, norm2);
    ggml_tensor * mlp   = ggml_mul(ctx, gate, up);
    ggml_tensor * down  = ggml_mul_mat(ctx, l.mlp_down, mlp);
    return ggml_add(ctx, inpFF, down);
}

// Build the shared cond_emb fragment: (n_embd, 34).
// exaggeration_tensor is a 1-D F32 tensor of length 1 with the emotion
// scalar (we multiply with emotion_adv_w to get the 1024-d emotion token).
ggml_tensor * build_cond_emb(ggml_context * ctx,
                             const chatterbox_model & m,
                             ggml_tensor * exaggeration) {
    const auto & hp = m.hparams;

    // 1. spkr_enc(speaker_emb): (n_embd, 1).
    //    cond_spkr_w ggml ne=(256, 1024) [from nn.Linear (out=1024, in=256) -> no
    //    explicit transpose, numpy <-> ggml axis reversal gives us (in, out)].
    //    builtin_speaker_emb ggml ne=(256, 1).  Result ne=(1024, 1).  Bias
    //    (1024,) broadcasts along the N=1 column.
    ggml_tensor * spkr_raw = ggml_mul_mat(ctx, m.cond_spkr_w, m.builtin_speaker_emb);
    ggml_tensor * spkr = ggml_add(ctx, spkr_raw,
                                   ggml_reshape_2d(ctx, m.cond_spkr_b, hp.n_embd, 1));

    // 2. cond_prompt_speech_emb = speech_emb[tokens] + speech_pos_emb[0..len).
    //    T3.prepare_conditioning adds positional embeddings to the speech
    //    tokens before handing them to the perceiver (not-is_gpt branch).
    ggml_tensor * cond_tok_emb = ggml_get_rows(ctx, m.speech_emb, m.builtin_cond_prompt_tokens);
    const int cond_prompt_len = m.builtin_cond_prompt_tokens->ne[0];
    ggml_tensor * cond_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cond_prompt_len);
    ggml_set_name(cond_pos_ids, "cond_prompt_pos_ids");
    ggml_set_input(cond_pos_ids);
    ggml_tensor * cond_pos = ggml_get_rows(ctx, m.speech_pos_emb, cond_pos_ids);
    ggml_tensor * cond_prompt_emb = ggml_add(ctx, cond_tok_emb, cond_pos);

    // 3. perceiver output: (n_embd, 32)
    ggml_tensor * perc = build_perceiver(ctx, m, cond_prompt_emb);

    // 4. emotion_adv: emotion_adv_w is (1, n_embd) after transpose; exaggeration
    //    is a (1,1) input scalar.  mul_mat((n_embd, 1), (1, 1)) → (n_embd, 1).
    //    Wait, emotion_adv_fc.weight in PyTorch is shape (1024, 1) (out, in).
    //    After transpose in the converter: (1, 1024) stored as ggml shape (1, 1024).
    //    mul_mat(w[K=1, M=1024], x[K=1, N=1]) → (M=1024, N=1).  Good.
    ggml_tensor * emot = ggml_mul_mat(ctx, m.emotion_adv_w, exaggeration);

    // 5. Concat along seq dim (ne[1]).  spkr(1024,1), perc(1024,32), emot(1024,1)
    //    → (1024, 34).
    ggml_tensor * cond_emb = ggml_concat(ctx, spkr, perc, /*dim=*/1);
    cond_emb = ggml_concat(ctx, cond_emb, emot, /*dim=*/1);
    return cond_emb;
}

// Build the prompt graph for either the conditional or unconditional pass.
//
//   tokens: the T_text text token IDs (same for both passes)
//   is_uncond: if true, zero out the text token embeddings (but keep text_pos_emb
//              and the BOS speech tokens unchanged).
ggml_cgraph * build_prompt_graph_mtl(const chatterbox_model & model,
                                     int n_text_tokens,
                                     bool is_uncond) {
    const auto & hp = model.hparams;
    const int len_cond = 1 + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0);
    const int N = len_cond + n_text_tokens + 2;  // +1 initial_speech, +1 bos

    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    // Dynamic inputs.
    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_tokens, "text_tokens");  ggml_set_input(text_tokens);

    ggml_tensor * speech_bos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_bos, "speech_bos");  ggml_set_input(speech_bos);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(pos_ids, "pos_ids");  ggml_set_input(pos_ids);

    ggml_tensor * text_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_pos_ids, "text_pos_ids");  ggml_set_input(text_pos_ids);

    ggml_tensor * speech_pos0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos0, "speech_pos0");  ggml_set_input(speech_pos0);

    ggml_tensor * exaggeration = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(exaggeration, "exaggeration");  ggml_set_input(exaggeration);

    // Causal attention mask for prompt path (N > 1).  F16 as required by Metal FA.
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, N);
    ggml_set_name(kq_mask, "kq_mask");  ggml_set_input(kq_mask);

    // 1. cond_emb (34 tokens).
    ggml_tensor * cond_emb = build_cond_emb(ctx, model, exaggeration);

    // 2. text_emb with learned pos (zeroed token part if uncond).
    ggml_tensor * text_pos_emb_seq = ggml_get_rows(ctx, model.text_pos_emb, text_pos_ids);
    ggml_tensor * text_emb_out;
    if (is_uncond) {
        text_emb_out = text_pos_emb_seq;
    } else {
        ggml_tensor * text_tok_emb = ggml_get_rows(ctx, model.text_emb, text_tokens);
        text_emb_out = ggml_add(ctx, text_tok_emb, text_pos_emb_seq);
    }

    // 3. Speech embeddings: initial_speech = bos (both are speech_emb(6561) + spos[0]).
    ggml_tensor * speech_tok_emb = ggml_get_rows(ctx, model.speech_emb, speech_bos);
    ggml_tensor * speech_pos_emb_0 = ggml_get_rows(ctx, model.speech_pos_emb, speech_pos0);
    ggml_tensor * speech_emb_out = ggml_add(ctx, speech_tok_emb, speech_pos_emb_0);

    // 4. Concat: cond_emb | text_emb | initial_speech | bos.
    ggml_tensor * inp = ggml_concat(ctx, cond_emb, text_emb_out, /*dim=*/1);
    inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);
    inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);

    // 5. Run 30 Llama layers.
    ggml_tensor * mem_k = is_uncond ? model.memory_k_uncond : model.memory_k;
    ggml_tensor * mem_v = is_uncond ? model.memory_v_uncond : model.memory_v;
    ggml_tensor * cur = inp;
    for (int il = 0; il < hp.n_layer; ++il) {
        cur = build_llama_block(ctx, gf, model, il, cur, /*n_past=*/0, N, mem_k, mem_v,
                                pos_ids, kq_mask);
    }

    // Final RMSNorm + speech_head (take logits at last position only — seq index N-1).
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), model.norm_g);
    // cur: (n_embd, N) -> take last column.
    ggml_tensor * last = ggml_view_2d(ctx, cur, hp.n_embd, 1,
                                      cur->nb[1],
                                      (size_t)(N - 1) * cur->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, model.speech_head, last);  // (n_speech_vocab, 1)
    ggml_set_name(logits, "logits");  ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_step_graph_mtl(const chatterbox_model & model,
                                   int n_past,
                                   bool is_uncond) {
    const auto & hp = model.hparams;

    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * speech_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_token, "speech_token"); ggml_set_input(speech_token);

    ggml_tensor * speech_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos, "speech_pos"); ggml_set_input(speech_pos);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    ggml_tensor * inp = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, speech_token),
        ggml_get_rows(ctx, model.speech_pos_emb, speech_pos));

    ggml_tensor * mem_k = is_uncond ? model.memory_k_uncond : model.memory_k;
    ggml_tensor * mem_v = is_uncond ? model.memory_v_uncond : model.memory_v;

    ggml_tensor * cur = inp;
    for (int il = 0; il < hp.n_layer; ++il) {
        cur = build_llama_block(ctx, gf, model, il, cur, n_past, /*N=*/1, mem_k, mem_v,
                                pos_ids, /*kq_mask=*/nullptr);
    }
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), model.norm_g);

    ggml_tensor * logits = ggml_mul_mat(ctx, model.speech_head, cur);  // (n_speech_vocab, 1)
    ggml_set_name(logits, "logits"); ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx);
    return gf;
}

void fill_causal_mask_f16(std::vector<ggml_fp16_t> & out, int N) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    out.assign((size_t) N * N, zero);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out[(size_t) i * N + j] = (j <= i) ? zero : neg_inf;
        }
    }
}

bool run_prompt_pass(const chatterbox_model & model,
                     ggml_gallocr_t allocr,
                     int n_threads,
                     const std::vector<int32_t> & text_tokens,
                     float exaggeration,
                     bool is_uncond,
                     std::vector<float> & logits_out,
                     int & prompt_len_out) {
    const auto & hp = model.hparams;
    const int len_cond = 1 + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0);
    const int N = len_cond + (int) text_tokens.size() + 2;
    prompt_len_out = N;

    ggml_cgraph * gf = build_prompt_graph_mtl(model, (int) text_tokens.size(), is_uncond);
    if (!ggml_gallocr_reserve(allocr, gf)) {
        fprintf(stderr, "run_prompt_pass: gallocr_reserve failed\n");
        return false;
    }
    ggml_gallocr_alloc_graph(allocr, gf);

    // Dynamic inputs.  Any tensor may be pruned by the allocator if it does
    // not feed into the final output (e.g. text_tokens is unused on the
    // uncond pass where text_emb is replaced by zeros), so null-check.
    auto set_in = [&](const char * name, const void * data, size_t bytes) {
        ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        if (t) ggml_backend_tensor_set(t, data, 0, bytes);
    };
    set_in("text_tokens", text_tokens.data(), text_tokens.size() * sizeof(int32_t));
    int32_t bos = hp.start_speech_token;
    set_in("speech_bos", &bos, sizeof(bos));

    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = i;
    set_in("pos_ids", pos.data(), pos.size() * sizeof(int32_t));

    std::vector<int32_t> text_pos(text_tokens.size());
    for (size_t i = 0; i < text_tokens.size(); ++i) text_pos[i] = (int32_t) i;
    set_in("text_pos_ids", text_pos.data(), text_pos.size() * sizeof(int32_t));

    int32_t sp0 = 0;
    set_in("speech_pos0", &sp0, sizeof(sp0));

    const int cond_prompt_len = hp.cond_prompt_len;
    std::vector<int32_t> cond_pos(cond_prompt_len);
    for (int i = 0; i < cond_prompt_len; ++i) cond_pos[i] = i;
    set_in("cond_prompt_pos_ids", cond_pos.data(), cond_pos.size() * sizeof(int32_t));

    float exag = exaggeration;
    set_in("exaggeration", &exag, sizeof(exag));

    // Causal mask.
    std::vector<ggml_fp16_t> mask;
    fill_causal_mask_f16(mask, N);
    set_in("kq_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(ggml_nelements(logits));
    ggml_backend_tensor_get(logits, logits_out.data(), 0, ggml_nbytes(logits));
    return true;
}

bool run_step_pass(const chatterbox_model & model,
                   ggml_gallocr_t allocr,
                   int n_threads,
                   int n_past,
                   int32_t token,
                   bool is_uncond,
                   std::vector<float> & logits_out) {
    ggml_cgraph * gf = build_step_graph_mtl(model, n_past, is_uncond);
    if (!ggml_gallocr_reserve(allocr, gf)) {
        fprintf(stderr, "run_step_pass: gallocr_reserve failed\n");
        return false;
    }
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &token, 0, sizeof(token));
    int32_t sp = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_pos"), &sp, 0, sizeof(sp));
    int32_t pos = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"), &pos, 0, sizeof(pos));

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(ggml_nelements(logits));
    ggml_backend_tensor_get(logits, logits_out.data(), 0, ggml_nbytes(logits));
    return true;
}

} // namespace

// -- Stage builders for parity validation (see t3_mtl.h) --------------------

ggml_cgraph * build_stage_cond_emb_graph(const chatterbox_model & m) {
    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * exag = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(exag, "exaggeration"); ggml_set_input(exag);

    ggml_tensor * out = build_cond_emb(ctx, m, exag);
    ggml_set_name(out, "cond_emb");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_stage_text_emb_graph(const chatterbox_model & m, int T_text) {
    static size_t buf_size = ggml_tensor_overhead() * 256 +
                             ggml_graph_overhead_custom(256, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_tokens, "text_tokens"); ggml_set_input(text_tokens);
    ggml_tensor * text_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_pos_ids, "text_pos_ids"); ggml_set_input(text_pos_ids);

    ggml_tensor * tok = ggml_get_rows(ctx, m.text_emb, text_tokens);
    ggml_tensor * pos = ggml_get_rows(ctx, m.text_pos_emb, text_pos_ids);
    ggml_tensor * out = ggml_add(ctx, tok, pos);
    ggml_set_name(out, "text_emb_with_pos");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_stage_inputs_graph(const chatterbox_model & m, int T_text,
                                       bool is_uncond) {
    const auto & hp = m.hparams;
    const int len_cond = 1 + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0);
    const int N = len_cond + T_text + 2;

    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_tokens, "text_tokens"); ggml_set_input(text_tokens);
    ggml_tensor * text_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_pos_ids, "text_pos_ids"); ggml_set_input(text_pos_ids);
    ggml_tensor * speech_bos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_bos, "speech_bos"); ggml_set_input(speech_bos);
    ggml_tensor * speech_pos0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos0, "speech_pos0"); ggml_set_input(speech_pos0);
    ggml_tensor * exag = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(exag, "exaggeration"); ggml_set_input(exag);

    ggml_tensor * cond_emb = build_cond_emb(ctx, m, exag);

    ggml_tensor * text_pos = ggml_get_rows(ctx, m.text_pos_emb, text_pos_ids);
    ggml_tensor * text_emb_out;
    if (is_uncond) {
        text_emb_out = text_pos;
    } else {
        ggml_tensor * tok = ggml_get_rows(ctx, m.text_emb, text_tokens);
        text_emb_out = ggml_add(ctx, tok, text_pos);
    }

    ggml_tensor * semb = ggml_get_rows(ctx, m.speech_emb, speech_bos);
    ggml_tensor * spos = ggml_get_rows(ctx, m.speech_pos_emb, speech_pos0);
    ggml_tensor * speech_emb_out = ggml_add(ctx, semb, spos);

    ggml_tensor * inp = ggml_concat(ctx, cond_emb, text_emb_out, /*dim=*/1);
    inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);
    inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);
    ggml_set_name(inp, "inputs_embeds");
    ggml_set_output(inp);
    ggml_build_forward_expand(gf, inp);
    (void) N;
    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_stage_layers_graph(const chatterbox_model & m, int N,
                                       int n_layers, bool is_uncond) {
    const auto & hp = m.hparams;
    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, N);
    ggml_set_name(inp, "inputs_embeds"); ggml_set_input(inp);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    ggml_tensor * kq_mask = nullptr;
    if (N > 1) {
        kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, N);
        ggml_set_name(kq_mask, "kq_mask"); ggml_set_input(kq_mask);
    }

    ggml_tensor * mem_k = is_uncond ? m.memory_k_uncond : m.memory_k;
    ggml_tensor * mem_v = is_uncond ? m.memory_v_uncond : m.memory_v;

    ggml_tensor * cur = inp;
    const int actual_layers = std::min(n_layers, hp.n_layer);
    for (int il = 0; il < actual_layers; ++il) {
        cur = build_llama_block(ctx, gf, m, il, cur, /*n_past=*/0, N,
                                mem_k, mem_v, pos_ids, kq_mask);
    }
    ggml_set_name(cur, "layers_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_stage_head_graph(const chatterbox_model & m, int N) {
    const auto & hp = m.hparams;
    static size_t buf_size = ggml_tensor_overhead() * 64 +
                             ggml_graph_overhead_custom(64, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, N);
    ggml_set_name(inp, "inputs_embeds"); ggml_set_input(inp);

    ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, inp, hp.eps), m.norm_g);
    ggml_tensor * last = ggml_view_2d(ctx, cur, hp.n_embd, 1,
                                      cur->nb[1],
                                      (size_t)(N - 1) * cur->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, m.speech_head, last);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    ggml_free(ctx);
    return gf;
}

// --------------------------------------------------------------------------

bool load_model_gguf_mtl(const std::string & path,
                         chatterbox_model & model,
                         int requested_ctx,
                         int n_gpu_layers) {
    extern int g_log_verbose;
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params params = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx) {
        fprintf(stderr, "load_model_gguf_mtl: failed to open '%s'\n", path.c_str());
        return false;
    }

    try {
        auto & hp = model.hparams;
        hp.variant           = CHBX_VARIANT_MTL;
        hp.n_text_vocab      = (int32_t) get_u32(gguf_ctx, KEY_TEXT_VOCAB_SIZE);
        hp.n_speech_vocab    = (int32_t) get_u32(gguf_ctx, KEY_SPEECH_VOCAB_SIZE);
        hp.start_speech_token = (int32_t) get_u32(gguf_ctx, KEY_START_SPEECH);
        hp.stop_speech_token  = (int32_t) get_u32(gguf_ctx, KEY_STOP_SPEECH);
        hp.start_text_token   = (int32_t) get_u32(gguf_ctx, KEY_START_TEXT);
        hp.stop_text_token    = (int32_t) get_u32(gguf_ctx, KEY_STOP_TEXT);
        hp.speaker_embed_size = (int32_t) get_u32(gguf_ctx, KEY_SPEAKER_EMBED);
        hp.cond_prompt_len    = (int32_t) get_u32(gguf_ctx, KEY_COND_PROMPT_LEN);
        hp.n_ctx   = (int32_t) get_u32(gguf_ctx, KEY_N_CTX);
        hp.n_embd  = (int32_t) get_u32(gguf_ctx, KEY_N_EMBD);
        hp.n_head  = (int32_t) get_u32(gguf_ctx, KEY_N_HEAD);
        hp.n_kv_head = (int32_t) get_u32(gguf_ctx, KEY_N_KV_HEAD);
        hp.head_dim = (int32_t) get_u32(gguf_ctx, KEY_HEAD_DIM);
        hp.intermediate_size = (int32_t) get_u32(gguf_ctx, KEY_INTERMEDIATE_SIZE);
        hp.n_layer = (int32_t) get_u32(gguf_ctx, KEY_N_LAYER);
        hp.max_text_tokens   = (int32_t) get_u32(gguf_ctx, KEY_MAX_TEXT_TOKENS);
        hp.max_speech_tokens = (int32_t) get_u32(gguf_ctx, KEY_MAX_SPEECH_TOKENS);
        hp.speech_cond_prompt_len = (int32_t) get_u32(gguf_ctx, KEY_SPEECH_COND_LEN);
        hp.perceiver_queries = (int32_t) get_u32(gguf_ctx, KEY_PERCEIVER_QUERIES);
        hp.perceiver_heads   = (int32_t) get_u32(gguf_ctx, KEY_PERCEIVER_HEADS);
        hp.emotion_adv       = get_bool(gguf_ctx, KEY_EMOTION_ADV);
        hp.eps               = get_f32(gguf_ctx, KEY_RMS_EPS);
        hp.rope_theta        = get_f32(gguf_ctx, KEY_ROPE_THETA);
        hp.rope_scale_factor = get_f32(gguf_ctx, KEY_ROPE_SCALING_FACTOR);
        hp.rope_low_freq     = get_f32(gguf_ctx, KEY_ROPE_LOW_FREQ);
        hp.rope_high_freq    = get_f32(gguf_ctx, KEY_ROPE_HIGH_FREQ);
        hp.rope_orig_max_pos = (int32_t) get_u32(gguf_ctx, KEY_ROPE_ORIG_MAX_POS);

        if (requested_ctx > 0) hp.n_ctx = std::min(hp.n_ctx, requested_ctx);

        model.backend = init_backend(n_gpu_layers);

        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        ggml_init_params p = { ggml_tensor_overhead() * (size_t)(num_tensors + 1), nullptr, true };
        model.ctx_w = ggml_init(p);
        if (!model.ctx_w) throw std::runtime_error("ggml_init failed");

        for (int64_t i = 0; i < num_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
            ggml_tensor * dst = ggml_dup_tensor(model.ctx_w, src);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;
        }

        const int half_hd = hp.head_dim / 2;
        ggml_tensor * freq_factors = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, half_hd);
        ggml_set_name(freq_factors, "rope_freq_factors");
        model.rope_freq_factors = freq_factors;

        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);

        for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w); cur; cur = ggml_get_next_tensor(model.ctx_w, cur)) {
            if (cur == freq_factors) continue;
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
            ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
        }

        {
            std::vector<float> ff = compute_llama3_freq_factors(
                hp.head_dim, hp.rope_theta, hp.rope_scale_factor,
                hp.rope_low_freq, hp.rope_high_freq, hp.rope_orig_max_pos);
            ggml_backend_tensor_set(freq_factors, ff.data(), 0, ff.size() * sizeof(float));
        }

        model.text_emb        = require_tensor(model, "chatterbox/text_emb");
        model.speech_emb      = require_tensor(model, "chatterbox/speech_emb");
        model.text_pos_emb    = require_tensor(model, "chatterbox/text_pos_emb");
        model.speech_pos_emb  = require_tensor(model, "chatterbox/speech_pos_emb");
        model.text_head       = require_tensor(model, "chatterbox/text_head");
        model.speech_head     = require_tensor(model, "chatterbox/speech_head");
        model.norm_g          = require_tensor(model, "model/norm/g");
        model.cond_spkr_w     = require_tensor(model, "chatterbox/cond_spkr/w");
        model.cond_spkr_b     = require_tensor(model, "chatterbox/cond_spkr/b");
        model.emotion_adv_w   = require_tensor(model, "chatterbox/emotion_adv_fc/w");

        model.perceiver.pre_attention_query = require_tensor(model, "chatterbox/perceiver/pre_attention_query");
        model.perceiver.norm_g = require_tensor(model, "chatterbox/perceiver/attn/norm/g");
        model.perceiver.norm_b = require_tensor(model, "chatterbox/perceiver/attn/norm/b");
        model.perceiver.to_q_w = require_tensor(model, "chatterbox/perceiver/attn/to_q/w");
        model.perceiver.to_q_b = require_tensor(model, "chatterbox/perceiver/attn/to_q/b");
        model.perceiver.to_k_w = require_tensor(model, "chatterbox/perceiver/attn/to_k/w");
        model.perceiver.to_k_b = require_tensor(model, "chatterbox/perceiver/attn/to_k/b");
        model.perceiver.to_v_w = require_tensor(model, "chatterbox/perceiver/attn/to_v/w");
        model.perceiver.to_v_b = require_tensor(model, "chatterbox/perceiver/attn/to_v/b");
        model.perceiver.proj_out_w = require_tensor(model, "chatterbox/perceiver/attn/proj_out/w");
        model.perceiver.proj_out_b = require_tensor(model, "chatterbox/perceiver/attn/proj_out/b");

        model.builtin_speaker_emb        = require_tensor(model, "chatterbox/builtin/speaker_emb");
        model.builtin_cond_prompt_tokens = require_tensor(model, "chatterbox/builtin/cond_prompt_speech_tokens");

        model.layers_mtl.resize(hp.n_layer);
        for (int i = 0; i < hp.n_layer; ++i) {
            auto & l = model.layers_mtl[i];
            std::string lp = "model/h" + std::to_string(i);
            l.ln_attn_g = require_tensor(model, (lp + "/ln_attn/g").c_str());
            l.ln_mlp_g  = require_tensor(model, (lp + "/ln_mlp/g").c_str());
            l.wq = require_tensor(model, (lp + "/attn/q/w").c_str());
            l.wk = require_tensor(model, (lp + "/attn/k/w").c_str());
            l.wv = require_tensor(model, (lp + "/attn/v/w").c_str());
            l.wo = require_tensor(model, (lp + "/attn/o/w").c_str());
            l.mlp_gate = require_tensor(model, (lp + "/mlp/gate/w").c_str());
            l.mlp_up   = require_tensor(model, (lp + "/mlp/up/w").c_str());
            l.mlp_down = require_tensor(model, (lp + "/mlp/down/w").c_str());
        }

        ggml_init_params kv_params = { ggml_tensor_overhead() * 4, nullptr, true };
        model.ctx_kv = ggml_init(kv_params);
        const int64_t kv_elements = (int64_t) hp.head_dim * hp.n_kv_head * hp.n_ctx * hp.n_layer;
        model.memory_k        = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, kv_elements);
        model.memory_v        = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, kv_elements);
        model.memory_k_uncond = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, kv_elements);
        model.memory_v_uncond = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, kv_elements);
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);

        {
            const int64_t jk = gguf_find_key(gguf_ctx, "tokenizer.ggml.mtl_json");
            const int64_t lk = gguf_find_key(gguf_ctx, "tokenizer.ggml.mtl_languages");
            if (jk < 0) {
                fprintf(stderr, "load_model_gguf_mtl: GGUF missing tokenizer.ggml.mtl_json; "
                                "re-run scripts/convert-t3-mtl-to-gguf.py.\n");
                gguf_free(gguf_ctx);
                ggml_free(tmp_ctx);
                return false;
            }
            model.mtl_tokenizer_json = gguf_get_val_str(gguf_ctx, jk);
            if (lk >= 0) {
                const size_t n = gguf_get_arr_n(gguf_ctx, lk);
                model.mtl_languages.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    model.mtl_languages.emplace_back(gguf_get_arr_str(gguf_ctx, lk, i));
                }
            }
        }

        if (g_log_verbose) {
            fprintf(stderr, "load_model_gguf_mtl: ctx=%d embd=%d layers=%d heads=%d kv_heads=%d "
                            "head_dim=%d inter=%d text_vocab=%d speech_vocab=%d cond_prompt=%d\n",
                    hp.n_ctx, hp.n_embd, hp.n_layer, hp.n_head, hp.n_kv_head,
                    hp.head_dim, hp.intermediate_size,
                    hp.n_text_vocab, hp.n_speech_vocab, hp.cond_prompt_len);
            fprintf(stderr, "load_model_gguf_mtl: weights=%.2f MB  KV=%.2f MB (2x for CFG) "
                            "tokenizer_json=%zu bytes  languages=%zu\n",
                    ggml_backend_buffer_get_size(model.buffer_w) / (1024.0*1024.0),
                    ggml_backend_buffer_get_size(model.buffer_kv) / (1024.0*1024.0),
                    model.mtl_tokenizer_json.size(), model.mtl_languages.size());
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "load_model_gguf_mtl: %s\n", e.what());
        gguf_free(gguf_ctx); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }

    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    return true;
}

bool eval_prompt_mtl(const chatterbox_model & model,
                     ggml_gallocr_t allocr,
                     int n_threads,
                     const std::vector<int32_t> & text_tokens,
                     float exaggeration,
                     std::vector<float> & logits_cond_out,
                     std::vector<float> & logits_uncond_out,
                     int & prompt_len) {
    int plen_c = 0, plen_u = 0;
    if (!run_prompt_pass(model, allocr, n_threads, text_tokens, exaggeration,
                         /*is_uncond=*/false, logits_cond_out, plen_c)) return false;
    if (!run_prompt_pass(model, allocr, n_threads, text_tokens, exaggeration,
                         /*is_uncond=*/true, logits_uncond_out, plen_u)) return false;
    prompt_len = plen_c;
    return plen_c == plen_u;
}

bool eval_step_mtl(const chatterbox_model & model,
                   ggml_gallocr_t allocr,
                   int n_threads,
                   int n_past,
                   int32_t token,
                   std::vector<float> & logits_cond_out,
                   std::vector<float> & logits_uncond_out) {
    if (!run_step_pass(model, allocr, n_threads, n_past, token, /*uncond=*/false,
                       logits_cond_out)) return false;
    if (!run_step_pass(model, allocr, n_threads, n_past, token, /*uncond=*/true,
                       logits_uncond_out)) return false;
    return true;
}

// Sampler (CFG + rep penalty + temperature + min_p + top_p).  Top-k clamp is
// optional (ignored if <= 0).  Mirrors the LogitsProcessorList order used by
// ChatterboxMultilingualTTS.generate.
int32_t sample_next_token_mtl(const std::vector<float> & logits_cond,
                              const std::vector<float> & logits_uncond,
                              const std::vector<int32_t> & generated,
                              const chatterbox_sampling_params & p,
                              std::mt19937 & rng) {
    const size_t V = logits_cond.size();
    std::vector<float> l(V);
    for (size_t i = 0; i < V; ++i) {
        l[i] = logits_cond[i] + p.cfg_weight * (logits_cond[i] - logits_uncond[i]);
    }

    if (p.repeat_penalty != 1.0f) {
        for (int32_t t : generated) {
            if (t < 0 || (size_t) t >= V) continue;
            if (l[t] > 0.0f) l[t] /= p.repeat_penalty;
            else             l[t] *= p.repeat_penalty;
        }
    }

    if (p.temp > 0.0f && p.temp != 1.0f) {
        for (float & x : l) x /= p.temp;
    }

    if (p.min_p > 0.0f) {
        float maxl = -INFINITY;
        for (float x : l) if (x > maxl) maxl = x;
        const float thresh = maxl + std::log(p.min_p);
        for (float & x : l) if (x < thresh) x = -INFINITY;
    }

    if (p.top_p < 1.0f && p.top_p > 0.0f) {
        std::vector<int> idx(V);
        for (size_t i = 0; i < V; ++i) idx[i] = (int) i;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return l[a] > l[b]; });
        float maxl = l[idx[0]];
        double sum = 0.0;
        std::vector<double> probs(V);
        for (size_t i = 0; i < V; ++i) {
            probs[i] = std::exp((double)(l[i] - maxl));
            sum += probs[i];
        }
        double cum = 0.0;
        for (size_t i = 0; i < V; ++i) {
            cum += probs[idx[i]] / sum;
            if (cum >= p.top_p) {
                for (size_t j = i + 1; j < V; ++j) l[idx[j]] = -INFINITY;
                break;
            }
        }
    }

    if (p.top_k > 0 && (size_t) p.top_k < V) {
        std::vector<int> idx(V);
        for (size_t i = 0; i < V; ++i) idx[i] = (int) i;
        std::nth_element(idx.begin(), idx.begin() + p.top_k, idx.end(),
                         [&](int a, int b){ return l[a] > l[b]; });
        const float cut = l[idx[p.top_k - 1]];
        for (float & x : l) if (x < cut) x = -INFINITY;
    }

    double maxl = -INFINITY;
    for (float x : l) if (x > maxl) maxl = x;
    if (!std::isfinite(maxl)) {
        return 0;
    }
    double sum = 0.0;
    std::vector<double> probs(V);
    for (size_t i = 0; i < V; ++i) {
        probs[i] = std::exp((double)(l[i] - maxl));
        sum += probs[i];
    }
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng);
    double cum = 0.0;
    for (size_t i = 0; i < V; ++i) {
        cum += probs[i];
        if (cum >= r) return (int32_t) i;
    }
    return (int32_t)(V - 1);
}

} // namespace tts_cpp::chatterbox::detail
