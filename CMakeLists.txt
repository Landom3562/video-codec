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
)

# Create encoder executable
add_executable(encoder src/encoder.c)
target_link_libraries(encoder PRIVATE PkgConfig::FFMPEG)
target_include_directories(encoder PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Create decoder executable
add_executable(decoder src/decoder.c)
target_link_libraries(decoder PRIVATE PkgConfig::FFMPEG)
target_include_directories(decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g")