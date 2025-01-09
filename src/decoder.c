#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include "codec_utils.h"

void decode(const char* input_filename, const char* output_filename) {
    AVFormatContext *input_ctx = NULL, *output_ctx = NULL;
    AVCodec *codec = NULL;
    AVCodecContext *codec_ctx = NULL;
    AVStream *in_stream = NULL, *out_stream = NULL;
    int ret, stream_idx = -1;

    // Open input file
    if (avformat_open_input(&input_ctx, input_filename, NULL, NULL) < 0) {
        handle_error("Could not open input file");
    }

    // Find stream info
    if (avformat_find_stream_info(input_ctx, NULL) < 0) {
        handle_error("Could not find stream info");
    }

    // Find video stream
    for (unsigned int i = 0; i < input_ctx->nb_streams; i++) {
        if (input_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            stream_idx = i;
            in_stream = input_ctx->streams[i];
            break;
        }
    }

    if (stream_idx == -1) {
        handle_error("Could not find video stream");
    }

    // Find decoder
    codec = avcodec_find_decoder(in_stream->codecpar->codec_id);
    if (!codec) {
        handle_error("Could not find decoder");
    }

    // Create decoder context
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        handle_error("Could not allocate decoder context");
    }

    // Fill decoder context
    if (avcodec_parameters_to_context(codec_ctx, in_stream->codecpar) < 0) {
        handle_error("Could not copy codec params");
    }

    // Initialize decoder
    if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
        handle_error("Could not open codec");
    }

    // Create output context
    avformat_alloc_output_context2(&output_ctx, NULL, NULL, output_filename);
    if (!output_ctx) {
        handle_error("Could not create output context");
    }

    // Create output stream
    out_stream = avformat_new_stream(output_ctx, codec);
    if (!out_stream) {
        handle_error("Could not create output stream");
    }

    // Copy stream parameters
    if (avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar) < 0) {
        handle_error("Could not copy stream params");
    }

    // Open output file
    if (!(output_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&output_ctx->pb, output_filename, AVIO_FLAG_WRITE) < 0) {
            handle_error("Could not open output file");
        }
    }

    // Write header
    if (avformat_write_header(output_ctx, NULL) < 0) {
        handle_error("Could not write header");
    }

    // Read and write packets
    AVPacket *pkt = av_packet_alloc();
    
    while (av_read_frame(input_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            // Rescale timestamps
            pkt->pts = av_rescale_q_rnd(pkt->pts,
                                      in_stream->time_base,
                                      out_stream->time_base,
                                      AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
            pkt->dts = av_rescale_q_rnd(pkt->dts,
                                      in_stream->time_base,
                                      out_stream->time_base,
                                      AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
            pkt->duration = av_rescale_q(pkt->duration,
                                      in_stream->time_base,
                                      out_stream->time_base);
            pkt->pos = -1;
            pkt->stream_index = 0;

            ret = av_interleaved_write_frame(output_ctx, pkt);
            if (ret < 0) {
                handle_error("Error writing frame");
            }
        }
        av_packet_unref(pkt);
    }

    // Write trailer
    av_write_trailer(output_ctx);

    // Cleanup
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&input_ctx);
    if (output_ctx && !(output_ctx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&output_ctx->pb);
    }
    avformat_free_context(output_ctx);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    
    decode(argv[1], argv[2]);
    return 0;
}