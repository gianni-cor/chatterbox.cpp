#include "voice_features.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// ============================================================================
// WAV I/O
// ============================================================================

bool wav_load(const std::string & path,
              std::vector<float> & out_samples,
              int & out_sr)
{
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        fprintf(stderr, "wav_load: failed to open %s\n", path.c_str());
        return false;
    }
    out_sr = (int)wav.sampleRate;

    std::vector<float> interleaved(wav.totalPCMFrameCount * wav.channels);
    drwav_uint64 frames = drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, interleaved.data());
    if (frames != wav.totalPCMFrameCount) {
        fprintf(stderr, "wav_load: short read (%llu / %llu)\n",
                (unsigned long long)frames, (unsigned long long)wav.totalPCMFrameCount);
    }

    // Down-mix to mono.
    out_samples.resize(frames);
    if (wav.channels == 1) {
        std::memcpy(out_samples.data(), interleaved.data(), frames * sizeof(float));
    } else {
        const int ch = (int)wav.channels;
        for (drwav_uint64 i = 0; i < frames; ++i) {
            float acc = 0.0f;
            for (int c = 0; c < ch; ++c) acc += interleaved[i * ch + c];
            out_samples[i] = acc / (float)ch;
        }
    }

    drwav_uninit(&wav);
    return true;
}

// ============================================================================
// Resampling (Kaiser-windowed sinc, rational ratio)
// ============================================================================

// Modified Bessel function I0(x), series summation sufficient for |x| < 50.
static double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double half = 0.5 * x;
    for (int k = 1; k < 30; ++k) {
        term *= (half / (double)k) * (half / (double)k);
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
}

// Greatest common divisor.
static int gcd_int(int a, int b) {
    while (b) { int t = a % b; a = b; b = t; }
    return a;
}

std::vector<float> resample_sinc(const std::vector<float> & in,
                                 int sr_in, int sr_out,
                                 int taps_half)
{
    if (sr_in == sr_out) return in;
    if (in.empty()) return {};
    (void)gcd_int;  // historical helper, no longer needed

    // Straight sinc interpolation: for each output sample at fractional input
    // position t, accumulate h((t - k)) * x[k] over a window of 2*taps_half+1
    // surrounding input indices.  Cutoff at min(sr_in, sr_out)/2 prevents
    // aliasing when downsampling and keeps upsampling bandlimited.
    const double fc  = 0.5 * std::min(sr_in, sr_out) / (double)sr_in; // fraction of input rate
    const double beta = 8.6;   // Kaiser, ~ -90 dB sidelobe
    const double inv_i0_beta = 1.0 / bessel_i0(beta);

    const double rate  = (double)sr_out / (double)sr_in;
    const size_t L_in  = in.size();
    const size_t L_out = (size_t)std::floor((double)L_in * rate);
    std::vector<float> out(L_out, 0.0f);

    for (size_t n = 0; n < L_out; ++n) {
        const double t_in  = (double)n / rate;                    // fractional input index
        const long long center = (long long)std::floor(t_in);
        const double frac  = t_in - (double)center;

        float acc = 0.0f;
        for (int k = -taps_half; k <= taps_half; ++k) {
            const long long idx = center + k;
            if (idx < 0 || idx >= (long long)L_in) continue;

            const double offset = frac - (double)k;               // distance in input-sample units
            const double sinc_arg = 2.0 * M_PI * fc * offset;
            const double sinc = (std::fabs(offset) < 1e-12)
                ? 1.0
                : std::sin(sinc_arg) / sinc_arg;
            const double wrel = offset / (double)taps_half;
            const double win  = (std::fabs(wrel) <= 1.0)
                ? bessel_i0(beta * std::sqrt(1.0 - wrel * wrel)) * inv_i0_beta
                : 0.0;
            acc += (float)(2.0 * fc * sinc * win) * in[(size_t)idx];
        }
        out[n] = acc;
    }
    return out;
}

// ============================================================================
// Mel extraction at 24 kHz, 80 channels (matches s3gen mel_spectrogram)
// ============================================================================

// Reflect-pad along the time axis.  For p_left / p_right > 0, a length-L signal
// becomes length (L + p_left + p_right) via PyTorch's "reflect" semantics, i.e.
// mirror without repeating the boundary sample.
static void reflect_pad_1d(const std::vector<float> & in, int p_left, int p_right,
                           std::vector<float> & out)
{
    const int L = (int)in.size();
    out.resize((size_t)(L + p_left + p_right));
    // Left reflection: x[p_left], x[p_left-1], ..., x[1]
    for (int i = 0; i < p_left; ++i) {
        int src = p_left - i;
        out[i] = (src >= 0 && src < L) ? in[src] : 0.0f;
    }
    std::memcpy(out.data() + p_left, in.data(), L * sizeof(float));
    // Right reflection: x[L-2], x[L-3], ..., x[L-1-p_right]
    for (int i = 0; i < p_right; ++i) {
        int src = L - 2 - i;
        out[(size_t)(L + p_left + i)] = (src >= 0 && src < L) ? in[src] : 0.0f;
    }
}

// Shared mel-spectrogram core. Handles:
//   - center mode: 0 = center=False (reflect-pad by (n_fft-hop)/2), 1 = center=True
//                  (reflect-pad by n_fft/2 each side, produces 1 + L/hop frames).
//   - power_exponent: 1.0 = magnitude, 2.0 = power spectrogram.
//   - log_floor > 0 means log-compress with clamp(x, log_floor); <= 0 means no log.
static std::vector<float> mel_extract_generic(
    const std::vector<float> & wav,
    const std::vector<float> & mel_filterbank,
    int n_fft, int hop, int win, int n_mels,
    int center_mode,        // 0 = center=False, 1 = center=True
    float power_exponent,   // 1.0 or 2.0
    float log_floor,        // > 0 → log-compress with clamp; <= 0 → no log
    bool transpose_to_T_M)  // true: return (T, M); false: return (M, T)
{
    const int F = n_fft / 2 + 1;
    if (mel_filterbank.size() != (size_t)(n_mels * F)) {
        fprintf(stderr,
            "mel_extract_generic: filterbank has %zu elements, expected %d (n_mels * F)\n",
            mel_filterbank.size(), n_mels * F);
        return {};
    }

    // Reflect-pad.  center=False → (n_fft - hop)/2 each side.
    // center=True  → n_fft/2 each side (librosa default, matches
    // voice_encoder.melspec._stft / torch.stft center=True).
    const int pad = (center_mode == 0) ? (n_fft - hop) / 2 : n_fft / 2;
    std::vector<float> padded;
    reflect_pad_1d(wav, pad, pad, padded);
    const int L = (int)padded.size();

    if (L < win) return {};
    const int T = (center_mode == 0)
        ? (L - win) / hop + 1                      // (L - win) / hop + 1
        : 1 + (int)wav.size() / hop;               // librosa invariant

    std::vector<float> hann(win);
    for (int n = 0; n < win; ++n)
        hann[n] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)n / (float)win));

    std::vector<float> cos_tbl((size_t)F * n_fft);
    std::vector<float> sin_tbl((size_t)F * n_fft);
    for (int k = 0; k < F; ++k) {
        for (int n = 0; n < n_fft; ++n) {
            double th = 2.0 * M_PI * (double)k * (double)n / (double)n_fft;
            cos_tbl[(size_t)k * n_fft + n] = (float)std::cos(th);
            sin_tbl[(size_t)k * n_fft + n] = (float)std::sin(th);
        }
    }

    std::vector<float> spec((size_t)F * T);
    std::vector<float> frame(win);
    for (int t = 0; t < T; ++t) {
        const float * x = padded.data() + t * hop;
        for (int n = 0; n < win; ++n) frame[n] = x[n] * hann[n];
        for (int k = 0; k < F; ++k) {
            const float * cs = cos_tbl.data() + (size_t)k * n_fft;
            const float * sn = sin_tbl.data() + (size_t)k * n_fft;
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < win; ++n) {
                re += frame[n] * cs[n];
                im -= frame[n] * sn[n];  // torch stft uses exp(-j...)
            }
            float mag = std::sqrt(re * re + im * im + 1e-9f);
            if (power_exponent == 2.0f) mag = mag * mag;
            else if (power_exponent != 1.0f) mag = std::pow(mag, power_exponent);
            spec[(size_t)k * T + t] = mag;
        }
    }

    std::vector<float> mel((size_t)n_mels * T);
    for (int m = 0; m < n_mels; ++m) {
        const float * fb_row = mel_filterbank.data() + (size_t)m * F;
        for (int t = 0; t < T; ++t) {
            float acc = 0.0f;
            for (int k = 0; k < F; ++k) acc += fb_row[k] * spec[(size_t)k * T + t];
            mel[(size_t)m * T + t] = acc;
        }
    }

    if (log_floor > 0.0f)
        for (float & v : mel) v = std::log(std::max(v, log_floor));

    if (!transpose_to_T_M) return mel;

    std::vector<float> out((size_t)T * n_mels);
    for (int m = 0; m < n_mels; ++m)
        for (int t = 0; t < T; ++t)
            out[(size_t)t * n_mels + m] = mel[(size_t)m * T + t];
    return out;
}

std::vector<float> mel_extract_24k_80(const std::vector<float> & wav_24k,
                                      const std::vector<float> & mel_filterbank)
{
    // center=False, magnitude (power_exp=1), log-compress with 1e-5 floor,
    // transpose to (T, 80).
    return mel_extract_generic(wav_24k, mel_filterbank,
        /*n_fft=*/1920, /*hop=*/480, /*win=*/1920, /*n_mels=*/80,
        /*center=*/0, /*power_exp=*/1.0f, /*log_floor=*/1e-5f,
        /*transpose=*/true);
}

std::vector<float> mel_extract_16k_40(const std::vector<float> & wav_16k,
                                      const std::vector<float> & mel_filterbank)
{
    // center=True (librosa stft default), POWER spectrogram (mel_power=2.0),
    // NO log (mel_type='amp', normalized_mels=False), transpose to (T, 40).
    return mel_extract_generic(wav_16k, mel_filterbank,
        /*n_fft=*/400, /*hop=*/160, /*win=*/400, /*n_mels=*/40,
        /*center=*/1, /*power_exp=*/2.0f, /*log_floor=*/-1.0f,
        /*transpose=*/true);
}
