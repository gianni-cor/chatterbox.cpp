// Validation harness for the streaming port (PROGRESS.md B1).
//
// Reads Python's streaming dumps produced by
// `scripts/dump-streaming-reference.py`, runs the C++ pipeline once per
// chunk with the matching `finalize` flag and token prefix, and compares
// per-chunk mel output against Python's `chunk_{k}_mels_new.npy`.
//
// Usage:
//   ./build/test-streaming  models/chatterbox-s3gen.gguf  /tmp/streaming_ref/

#include "s3gen_pipeline.h"
#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <sys/stat.h>

static bool path_exists(const std::string & p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

struct cmp_result {
    double max_abs = 0, rms = 0, rel = 0;
    int n = 0;
};

static cmp_result cmp_f32(const float * a, const float * b, size_t n) {
    cmp_result r; r.n = (int)n;
    double ss = 0, max_ref = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > r.max_abs) r.max_abs = d;
        ss += d * d;
        double e = std::fabs((double)b[i]);
        if (e > max_ref) max_ref = e;
    }
    r.rms = std::sqrt(ss / n);
    r.rel = max_ref > 0 ? r.max_abs / max_ref : r.max_abs;
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s S3GEN.gguf STREAMING_REF_DIR\n", argv[0]);
        return 1;
    }
    const std::string gguf = argv[1];
    const std::string ref  = argv[2];

    npy_array all_tokens = npy_load(ref + "/speech_tokens.npy");
    const int n_speech = (int)all_tokens.n_elements();
    const int32_t * tok_ptr = (const int32_t *)all_tokens.data.data();
    fprintf(stderr, "loaded %d speech tokens\n", n_speech);

    int k = 0;
    int prev_mels_emitted = 0;
    double max_rel = 0;
    double total_rms = 0;
    while (true) {
        char kbuf[16];
        std::snprintf(kbuf, sizeof(kbuf), "%02d", k + 1);
        const std::string tok_path = ref + "/chunk_" + kbuf + "_tokens.npy";
        const std::string mel_path = ref + "/chunk_" + kbuf + "_mels_new.npy";
        if (!path_exists(tok_path)) break;

        npy_array toks_chunk = npy_load(tok_path);
        const int32_t * cp = (const int32_t *)toks_chunk.data.data();
        const int n_cum = (int)toks_chunk.n_elements();
        const bool is_last = (n_cum == n_speech);

        fprintf(stderr, "\n=== chunk %d ===  tokens_total=%d  finalize=%s  "
                        "prev_mels_emitted=%d\n",
                k + 1, n_cum, is_last ? "true" : "false", prev_mels_emitted);

        // Sanity-check the cumulative token prefix matches the global sequence.
        for (int i = 0; i < n_cum; ++i) {
            if (cp[i] != tok_ptr[i]) {
                fprintf(stderr, "error: chunk tokens diverge from global sequence at i=%d\n", i);
                return 1;
            }
        }

        std::vector<int32_t> chunk_tokens(cp, cp + n_cum);

        const std::string dump_path = ref + "/cpp_chunk_" + kbuf + "_mels_new.npy";

        s3gen_synthesize_opts opts;
        opts.s3gen_gguf_path           = gguf;
        opts.out_wav_path              = ref + "/cpp_chunk_" + kbuf + ".wav";
        opts.ref_dir                   = ref;
        opts.seed                      = 42;
        // Streaming semantics mirrored from scripts/dump-streaming-reference.py:
        //   * the Python loop appends 3 silence tokens ONCE at the top,
        //     before chunking, so per-chunk calls don't re-append.  We match
        //     that with append_lookahead_silence=false.
        //   * `finalize` only controls the tail 6-frame trim.
        //   * `skip_mel_frames` drops the frames already emitted by earlier
        //     chunks, leaving only what's new this chunk.
        opts.append_lookahead_silence  = false;
        opts.finalize                  = is_last;
        opts.skip_mel_frames           = prev_mels_emitted;
        opts.dump_mel_path             = dump_path;

        // Inject Python's exact per-chunk CFM noise so the two pipelines
        // become bit-exact comparable (bypasses the torch.randn vs
        // std::mt19937 divergence).
        // Use chunk_KK_step0_x_in.npy (the ACTUAL z passed into estimator.forward
        // at step 0 — includes the `noised_mels` overlay from flow_inference).
        // The older chunk_KK_cfm_z.npy only captured the first torch.randn_like
        // call and missed the meanflow-specific speech-region overwrite, which
        // caused post-prompt divergence.
        const std::string z_path = ref + "/chunk_" + kbuf + "_step0_x_in.npy";
        if (path_exists(z_path)) {
            npy_array z = npy_load(z_path);
            const float * zp = (const float *)z.data.data();
            opts.cfm_z0_override.assign(zp, zp + z.n_elements());
        } else {
            fprintf(stderr, "  (no cfm_z0 dump; using std::mt19937 — expect rel ≈ 0.25)\n");
        }

        int rc = s3gen_synthesize_to_wav(chunk_tokens, opts);
        if (rc != 0) {
            fprintf(stderr, "error: s3gen_synthesize_to_wav returned %d\n", rc);
            return rc;
        }

        // Compare mel against Python.
        npy_array py_mel  = npy_load(mel_path);
        npy_array cpp_mel = npy_load(dump_path);
        const float * py  = (const float *)py_mel.data.data();
        const float * cpp = (const float *)cpp_mel.data.data();
        const size_t n    = std::min(py_mel.n_elements(), cpp_mel.n_elements());
        const size_t py_rows  = py_mel.shape[0];
        const size_t cpp_rows = cpp_mel.shape[0];
        fprintf(stderr, "  mel shapes:   py=(%zu, %lld)  cpp=(%zu, %lld)\n",
                py_rows,  (long long)py_mel.shape[1],
                cpp_rows, (long long)cpp_mel.shape[1]);
        if (py_rows != cpp_rows) {
            fprintf(stderr, "  *** SHAPE MISMATCH — C++ emitted %zu frames, Python %zu\n",
                    cpp_rows, py_rows);
        }
        auto s = cmp_f32(cpp, py, n);
        fprintf(stderr, "  mel diff:     max_abs=%.4e  rms=%.4e  rel=%.4e\n",
                s.max_abs, s.rms, s.rel);
        if (s.rel > max_rel) max_rel = s.rel;
        total_rms += s.rms;
        prev_mels_emitted += (int)cpp_rows;
        k += 1;
    }

    fprintf(stderr, "\nresult: %d chunks verified, worst rel=%.4e, mean rms=%.4e\n",
            k, max_rel, total_rms / std::max(k, 1));
    return 0;
}
