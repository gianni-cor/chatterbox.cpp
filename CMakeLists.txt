cmake_minimum_required(VERSION 3.14)
project(tts-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TTS_CPP_STANDALONE_DEFAULT ON)
else()
    set(TTS_CPP_STANDALONE_DEFAULT OFF)
endif()

option(TTS_CPP_BUILD_LIBRARY     "tts-cpp: build the static tts-cpp library"                 ON)
option(TTS_CPP_BUILD_EXECUTABLES "tts-cpp: build the tts-cpp CLI + mel2wav demo"             ${TTS_CPP_STANDALONE_DEFAULT})
option(TTS_CPP_BUILD_TESTS       "tts-cpp: build per-stage numerical validation harnesses"   ${TTS_CPP_STANDALONE_DEFAULT})
option(TTS_CPP_INSTALL           "tts-cpp: generate install rules"                           ON)
option(TTS_CPP_USE_SYSTEM_GGML   "tts-cpp: use system-installed GGML library"                OFF)

if (NOT TARGET ggml)
    if (TTS_CPP_USE_SYSTEM_GGML)
        find_package(ggml CONFIG REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        add_subdirectory(ggml)
    endif()
endif()

find_package(OpenMP)

set(TTS_CPP_LIB_SOURCES
    src/main.cpp
    src/gpt2_bpe.cpp
    src/chatterbox_tts.cpp
    src/chatterbox_engine.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp
)

if (TTS_CPP_BUILD_LIBRARY)
    add_library(tts-cpp STATIC ${TTS_CPP_LIB_SOURCES})
    add_library(tts-cpp::tts-cpp ALIAS tts-cpp)

    target_include_directories(tts-cpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    target_link_libraries(tts-cpp PUBLIC ggml)

    if (OpenMP_CXX_FOUND)
        target_link_libraries(tts-cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if (GGML_CUDA)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_CUDA)
    endif()
    if (GGML_METAL)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_METAL)
    endif()
    if (GGML_VULKAN)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_VULKAN)
    endif()

    set_target_properties(tts-cpp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME tts-cpp
    )
endif()

if (TTS_CPP_BUILD_EXECUTABLES)
    if (NOT TTS_CPP_BUILD_LIBRARY)
        message(FATAL_ERROR "TTS_CPP_BUILD_EXECUTABLES requires TTS_CPP_BUILD_LIBRARY=ON")
    endif()

    add_executable(tts-cpp-cli src/cli_main.cpp)
    target_link_libraries(tts-cpp-cli PRIVATE tts-cpp)
    set_target_properties(tts-cpp-cli PROPERTIES OUTPUT_NAME tts-cpp)

    add_executable(mel2wav src/mel2wav.cpp)
    target_link_libraries(mel2wav PRIVATE ggml)
    target_include_directories(mel2wav PRIVATE ggml/include src)
endif()

if (TTS_CPP_BUILD_TESTS)
    add_executable(test-s3gen src/test_s3gen.cpp)
    target_link_libraries(test-s3gen PRIVATE ggml)
    target_include_directories(test-s3gen PRIVATE ggml/include src)

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

    add_executable(test-campplus
        src/test_campplus.cpp
        src/campplus.cpp)
    target_link_libraries(test-campplus PRIVATE ggml)
    target_include_directories(test-campplus PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-campplus PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(test-fbank
        src/test_fbank.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-fbank PRIVATE ggml)
    target_include_directories(test-fbank PRIVATE ggml/include src)

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

    add_executable(test-s3tokenizer
        src/test_s3tokenizer.cpp
        src/s3tokenizer.cpp)
    target_link_libraries(test-s3tokenizer PRIVATE ggml)
    target_include_directories(test-s3tokenizer PRIVATE ggml/include src)

    add_executable(test-streaming
        src/test_streaming.cpp
        src/chatterbox_tts.cpp)
    target_link_libraries(test-streaming PRIVATE ggml)
    target_include_directories(test-streaming PRIVATE ggml/include src include)

    add_executable(test-metal-ops src/test_metal_ops.cpp)
    target_link_libraries(test-metal-ops PRIVATE ggml)
    target_include_directories(test-metal-ops PRIVATE ggml/include src)
    if (GGML_METAL)
        target_compile_definitions(test-metal-ops PRIVATE GGML_USE_METAL)
    endif()
endif()

if (TTS_CPP_INSTALL AND TTS_CPP_BUILD_LIBRARY)
    install(TARGETS tts-cpp
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/tts-cpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tts-cppConfig.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tts-cppConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/tts-cpp
    )
endif()
