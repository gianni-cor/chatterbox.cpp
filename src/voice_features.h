#pragma once

// Voice-cloning preprocessing primitives: WAV I/O, resampling, and mel
// extraction. These replace the Python side of prepare-voice.py one piece at
// a time.
//
// Phase 1 status:
//   - wav_load                        done
//   - resample_sinc (Kaiser window)   done
//   - mel_extract_24k_80              done — produces prompt_feat (S3Gen side)
//
// The other four voice-cloning tensors (speaker_emb, cond_prompt_speech_tokens,
// embedding, prompt_token) still need VoiceEncoder / CAMPPlus / S3TokenizerV2
// ports and are covered by Phases 2-4 in PROGRESS.md.

#include <cstdint>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// WAV I/O
// -----------------------------------------------------------------------------

// Load a WAV file into a mono float32 buffer in [-1, 1].  Stereo / multi-channel
// inputs are averaged down to mono.  Returns false on IO or format errors.
// out_sr is the file's sample rate (may require resampling before use).
bool wav_load(const std::string & path,
              std::vector<float> & out_samples,
              int & out_sr);

// -----------------------------------------------------------------------------
// Resampling (rational-ratio windowed-sinc, Kaiser window, beta = 8.6 ~= -90 dB)
// -----------------------------------------------------------------------------

// Resample `in` from `sr_in` to `sr_out` using Kaiser-windowed sinc
// interpolation.  Quality is comparable to librosa's `res_type='kaiser_fast'`
// (32 taps) — good enough for voice preprocessing, not optimised for real-time.
std::vector<float> resample_sinc(const std::vector<float> & in,
                                 int sr_in, int sr_out,
                                 int taps_half = 16);

// -----------------------------------------------------------------------------
// Mel extraction
// -----------------------------------------------------------------------------

// Compute the 80-channel log-mel spectrogram at 24 kHz that S3Gen uses as
// `prompt_feat`, matching chatterbox.models.s3gen.utils.mel.mel_spectrogram:
//
//   n_fft=1920  hop=480  win=1920  fmin=0  fmax=8000  center=False
//   reflect-pad by (n_fft - hop) / 2 = 720 each side
//   magnitude (with 1e-9 floor), mel filterbank matmul, log(clip(x, 1e-5))
//
// `mel_filterbank` must be the (80 * 961 = 76880)-element filterbank that
// librosa.filters.mel produces for those parameters, flattened row-major (80
// rows of 961 columns).  It gets baked into the s3gen GGUF by
// convert-s3gen-to-gguf.py.
//
// Returns a row-major (T_mel, 80) tensor, where T_mel = (L_wav + 2*720 - 1920) / 480 + 1.
std::vector<float> mel_extract_24k_80(const std::vector<float> & wav_24k,
                                      const std::vector<float> & mel_filterbank);

// Compute the 40-channel mel *power* spectrogram at 16 kHz that VoiceEncoder
// consumes, matching voice_encoder.melspec.melspectrogram with its default
// hyperparameters:
//
//   n_fft=400  hop=160  win=400  center=True (reflect-pad n_fft/2 each side)
//   mel_power = 2 (POWER, not magnitude)
//   mel_type = "amp" (no log / db conversion)
//   normalized_mels = False
//
// `mel_filterbank` must be the librosa (40, 201)=8040-element filterbank
// produced with those parameters; convert-t3-turbo-to-gguf.py bakes it in as
// `voice_encoder/mel_fb`.
//
// Returns a row-major (T_mel, 40) tensor, where T_mel = 1 + L_wav / hop.
std::vector<float> mel_extract_16k_40(const std::vector<float> & wav_16k,
                                      const std::vector<float> & mel_filterbank);
