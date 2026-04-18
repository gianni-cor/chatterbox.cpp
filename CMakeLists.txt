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
    src/voice_features.cpp)
target_link_libraries(chatterbox PRIVATE ggml)
target_include_directories(chatterbox PRIVATE ggml/include src)

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
    src/voice_features.cpp)
target_link_libraries(test-voice-features PRIVATE ggml)
target_include_directories(test-voice-features PRIVATE ggml/include src)

add_executable(test-resample
    src/test_resample.cpp
    src/voice_features.cpp)
target_link_libraries(test-resample PRIVATE ggml)
target_include_directories(test-resample PRIVATE src)

if (GGML_CUDA)
    target_compile_definitions(chatterbox PRIVATE GGML_USE_CUDA)
endif()

if (GGML_METAL)
    target_compile_definitions(chatterbox PRIVATE GGML_USE_METAL)
endif()
