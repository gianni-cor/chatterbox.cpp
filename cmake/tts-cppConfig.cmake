include(CMakeFindDependencyMacro)
find_dependency(ggml CONFIG)

get_filename_component(_TTS_CPP_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

find_library(TTS_CPP_LIBRARY
    NAMES tts-cpp
    PATHS "${_TTS_CPP_PREFIX}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(TTS_CPP_INCLUDE_DIR
    NAMES tts-cpp/tts-cpp.h
    PATHS "${_TTS_CPP_PREFIX}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TARGET tts-cpp::tts-cpp)
    add_library(tts-cpp::tts-cpp STATIC IMPORTED)
    set_target_properties(tts-cpp::tts-cpp PROPERTIES
        IMPORTED_LOCATION             "${TTS_CPP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TTS_CPP_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES      "ggml::ggml"
    )
endif()

unset(_TTS_CPP_PREFIX)
