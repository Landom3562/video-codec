cmake_minimum_required(VERSION 3.10)
project(video_codec C)

# Find required FFmpeg packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
)

# Set source files
set(SOURCES
    src/encoder.c
    src/decoder.c
    src/codec_utils.c
)

# Create encoder executable
add_executable(encoder src/encoder.c src/codec_utils.c)
target_link_libraries(encoder PRIVATE PkgConfig::FFMPEG)
target_include_directories(encoder PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Create decoder executable
add_executable(decoder src/decoder.c src/codec_utils.c)
target_link_libraries(decoder PRIVATE PkgConfig::FFMPEG)
target_include_directories(decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Add tests if building with tests enabled
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    add_executable(test_encoder tests/test_encoder.c)
    add_executable(test_decoder tests/test_decoder.c)
    target_link_libraries(test_encoder PRIVATE PkgConfig::FFMPEG)
    target_link_libraries(test_decoder PRIVATE PkgConfig::FFMPEG)
endif()

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g")