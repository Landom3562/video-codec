#ifndef CODEC_UTILS_H
#define CODEC_UTILS_H

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

void handle_error(const char* message);
FILE* open_file(const char* filename, const char* mode);
void close_file(FILE* file);
void allocate_frame(AVFrame** frame, int width, int height);
void configure_codec_context(AVCodecContext* codec_ctx, int width, int height, int bitrate);
void write_frame_to_file(FILE* file, AVFrame* frame, int width, int height);
void read_frame_from_file(FILE* file, AVFrame* frame, int width, int height);

#endif