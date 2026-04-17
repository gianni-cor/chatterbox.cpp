cmake_minimum_required(VERSION 3.14)
project(qvac-chatterbox.cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml as a pristine vendored dependency
add_subdirectory(ggml)

add_executable(chatterbox src/main.cpp src/gpt2_bpe.cpp)
target_link_libraries(chatterbox PRIVATE ggml)
target_include_directories(chatterbox PRIVATE ggml/include src)

add_executable(test-s3gen src/test_s3gen.cpp)
target_link_libraries(test-s3gen PRIVATE ggml)
target_include_directories(test-s3gen PRIVATE ggml/include src)

add_executable(mel2wav src/mel2wav.cpp)
target_link_libraries(mel2wav PRIVATE ggml)
target_include_directories(mel2wav PRIVATE ggml/include src)

if (GGML_CUDA)
    target_compile_definitions(chatterbox PRIVATE GGML_USE_CUDA)
endif()

if (GGML_METAL)
    target_compile_definitions(chatterbox PRIVATE GGML_USE_METAL)
endif()
