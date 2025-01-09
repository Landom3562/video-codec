#include <stdio.h>
#include <stdlib.h>
#include "codec_utils.h"

void handle_error(const char *message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(EXIT_FAILURE);
}

FILE* open_file(const char *filename, const char *mode) {
    FILE *file = fopen(filename, mode);
    if (!file) {
        handle_error("Could not open file");
    }
    return file;
}

void close_file(FILE *file) {
    if (file) {
        fclose(file);
    }
}

void allocate_frame(AVFrame **frame, int width, int height) {
    *frame = av_frame_alloc();
    if (!*frame) {
        handle_error("Could not allocate frame");
    }
    (*frame)->width = width;
    (*frame)->height = height;
    (*frame)->format = AV_PIX_FMT_YUV420P;
    if (av_frame_get_buffer(*frame, 32) < 0) {
        handle_error("Could not allocate frame buffer");
    }
}

void configure_codec_context(AVCodecContext *codec_ctx, int width, int height, int bitrate) {
    codec_ctx->width = width;
    codec_ctx->height = height;
    codec_ctx->bit_rate = bitrate;
    codec_ctx->gop_size = 10;
    codec_ctx->max_b_frames = 1;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->time_base = (AVRational){1, 25};  // 25 fps
    codec_ctx->framerate = (AVRational){25, 1};

    // Set H.264 preset
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "preset", "medium", 0);
    av_dict_set(&opts, "tune", "zerolatency", 0);
    
    if (avcodec_open2(codec_ctx, codec_ctx->codec, &opts) < 0) {
        handle_error("Could not open codec");
    }
    av_dict_free(&opts);
}

void write_frame_to_file(FILE *file, AVFrame *frame, int width, int height) {
    // Write Y plane
    fwrite(frame->data[0], 1, width * height, file);
    
    // Write U plane
    fwrite(frame->data[1], 1, width * height / 4, file);
    
    // Write V plane
    fwrite(frame->data[2], 1, width * height / 4, file);
}

void read_frame_from_file(FILE *file, AVFrame *frame, int width, int height) {
    // Read Y plane
    if (fread(frame->data[0], 1, width * height, file) != width * height) {
        if (feof(file)) return;
        handle_error("Error reading Y plane");
    }
    
    // Read U plane
    if (fread(frame->data[1], 1, width * height / 4, file) != width * height / 4) {
        handle_error("Error reading U plane");
    }
    
    // Read V plane
    if (fread(frame->data[2], 1, width * height / 4, file) != width * height / 4) {
        handle_error("Error reading V plane");
    }
    
    frame->linesize[0] = width;
    frame->linesize[1] = width / 2;
    frame->linesize[2] = width / 2;
}