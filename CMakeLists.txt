cmake_minimum_required(VERSION 3.14)
project(qvac-chatterbox.cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml as a pristine vendored dependency
add_subdirectory(ggml)

# Unified end-to-end binary: text -> speech tokens (T3) -> wav (S3Gen + HiFT).
# Sources: T3 runner (main.cpp) + BPE tokenizer + S3Gen/HiFT pipeline
# (chatterbox_tts.cpp, wrapped as the s3gen_synthesize_to_wav entry point).
add_executable(chatterbox
    src/main.cpp
    src/gpt2_bpe.cpp
    src/chatterbox_tts.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp)
target_link_libraries(chatterbox PRIVATE ggml)
target_include_directories(chatterbox PRIVATE ggml/include src)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(chatterbox PRIVATE OpenMP::OpenMP_CXX)
endif()

# Staged numerical validation harness (optional).
add_executable(test-s3gen src/test_s3gen.cpp)
target_link_libraries(test-s3gen PRIVATE ggml)
target_include_directories(test-s3gen PRIVATE ggml/include src)

# Mel -> wav demo (HiFT vocoder in isolation; optional).
add_executable(mel2wav src/mel2wav.cpp)
target_link_libraries(mel2wav PRIVATE ggml)
target_include_directories(mel2wav PRIVATE ggml/include src)

# Voice-cloning preprocessing validation (WAV I/O + resample + 80-ch mel).
add_executable(test-voice-features
    src/test_voice_features.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp)
target_link_libraries(test-voice-features PRIVATE ggml)
target_include_directories(test-voice-features PRIVATE ggml/include src)

add_executable(test-resample
    src/test_resample.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp)
target_link_libraries(test-resample PRIVATE ggml)
target_include_directories(test-resample PRIVATE src)

add_executable(test-voice-encoder
    src/test_voice_encoder.cpp
    src/voice_encoder.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp)
target_link_libraries(test-voice-encoder PRIVATE ggml)
target_include_directories(test-voice-encoder PRIVATE ggml/include src)

# CAMPPlus validation harness (compares C++ forward against Python embedding).
add_executable(test-campplus
    src/test_campplus.cpp
    src/campplus.cpp)
target_link_libraries(test-campplus PRIVATE ggml)
target_include_directories(test-campplus PRIVATE ggml/include src)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(test-campplus PRIVATE OpenMP::OpenMP_CXX)
endif()

# Kaldi-style fbank validation harness.
add_executable(test-fbank
    src/test_fbank.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp)
target_link_libraries(test-fbank PRIVATE ggml)
target_include_directories(test-fbank PRIVATE ggml/include src)

# End-to-end wav → 192-d speaker embedding validation (CAMPPlus + fbank).
add_executable(test-voice-embedding
    src/test_voice_embedding.cpp
    src/campplus.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp)
target_link_libraries(test-voice-embedding PRIVATE ggml)
target_include_directories(test-voice-embedding PRIVATE ggml/include src)
if (OpenMP_CXX_FOUND)
    target_link_libraries(test-voice-embedding PRIVATE OpenMP::OpenMP_CXX)
endif()

# S3TokenizerV2 (speech-to-token encoder) validation harness.
add_executable(test-s3tokenizer
    src/test_s3tokenizer.cpp
    src/s3tokenizer.cpp)
target_link_libraries(test-s3tokenizer PRIVATE ggml)
target_include_directories(test-s3tokenizer PRIVATE ggml/include src)

# Streaming port (PROGRESS.md B1): per-chunk mel comparison against Python dumps.
add_executable(test-streaming
    src/test_streaming.cpp
    src/chatterbox_tts.cpp)
target_link_libraries(test-streaming PRIVATE ggml)
target_include_directories(test-streaming PRIVATE ggml/include src)

# Metal kernel validation: diag_mask_inf + pad with front padding.
add_executable(test-metal-ops src/test_metal_ops.cpp)
target_link_libraries(test-metal-ops PRIVATE ggml)
target_include_directories(test-metal-ops PRIVATE ggml/include src)
if (GGML_METAL)
    target_compile_definitions(test-metal-ops PRIVATE GGML_USE_METAL)
endif()

if (GGML_CUDA)
    target_compile_definitions(chatterbox PRIVATE GGML_USE_CUDA)
endif()

if (GGML_METAL)
    target_compile_definitions(chatterbox PRIVATE GGML_USE_METAL)
endif()

if (GGML_VULKAN)
    target_compile_definitions(chatterbox PRIVATE GGML_USE_VULKAN)
endif()
