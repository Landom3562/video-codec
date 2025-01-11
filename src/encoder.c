#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

void handle_error(const char *message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(EXIT_FAILURE);
}

void encode(const char* input_filename, const char* output_filename) {
    AVFormatContext *input_ctx = NULL, *output_ctx = NULL;
    AVStream *in_stream = NULL, *out_stream = NULL;
    AVCodec *codec = NULL;
    AVCodecContext *codec_ctx = NULL;
    AVFrame *frame = NULL;
    AVPacket *pkt = NULL;
    struct SwsContext *sws_ctx = NULL;
    int ret, stream_idx = -1;

    // Open input file
    if (avformat_open_input(&input_ctx, input_filename, NULL, NULL) < 0) {
        handle_error("Could not open input file");
    }

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

    // Create output context
    avformat_alloc_output_context2(&output_ctx, NULL, NULL, output_filename);
    if (!output_ctx) {
        handle_error("Could not create output context");
    }

    // Find encoder
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        handle_error("Could not find encoder");
    }

    // Create output stream
    out_stream = avformat_new_stream(output_ctx, NULL);
    if (!out_stream) {
        handle_error("Could not create output stream");
    }

    // Create encoder context
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        handle_error("Could not allocate encoder context");
    }

    // Set encoder parameters
    codec_ctx->height = in_stream->codecpar->height;
    codec_ctx->width = in_stream->codecpar->width;
    codec_ctx->sample_aspect_ratio = in_stream->codecpar->sample_aspect_ratio;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->time_base = (AVRational){1, 30};
    codec_ctx->framerate = (AVRational){30, 1};

    // Improve compression settings with better quality
    codec_ctx->bit_rate = 800000;  // Increase bitrate to 800Kbps for better quality
    codec_ctx->gop_size = 120;     // Reduce GOP size for better quality
    codec_ctx->max_b_frames = 2;    // Reduce B-frames slightly
    codec_ctx->flags |= AV_CODEC_FLAG_CLOSED_GOP;

    // Configure rate control
    codec_ctx->rc_min_rate = codec_ctx->bit_rate;
    codec_ctx->rc_max_rate = codec_ctx->bit_rate;
    codec_ctx->rc_buffer_size = codec_ctx->bit_rate * 2;

    // Set H.264 specific options for better quality while maintaining compression
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "preset", "medium", 0);     // Balance between speed and quality
    av_dict_set(&opts, "tune", "film", 0);         // Optimize for video content
    av_dict_set(&opts, "crf", "23", 0);            // Lower CRF for better quality (default is 23)
    av_dict_set(&opts, "profile", "main", 0);      // Use main profile for better quality
    av_dict_set(&opts, "level", "3.1", 0);         // Slightly higher level

    // Open encoder
    if (avcodec_open2(codec_ctx, codec, &opts) < 0) {
        handle_error("Could not open encoder");
    }

    // Copy encoder parameters to stream
    if (avcodec_parameters_from_context(out_stream->codecpar, codec_ctx) < 0) {
        handle_error("Could not copy encoder parameters");
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

    // Allocate frame and packet
    frame = av_frame_alloc();
    pkt = av_packet_alloc();
    if (!frame || !pkt) {
        handle_error("Could not allocate frame or packet");
    }

    // Set up frame
    frame->format = codec_ctx->pix_fmt;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;
    if (av_frame_get_buffer(frame, 32) < 0) {
        handle_error("Could not allocate frame data");
    }

    // Initialize input decoder
    AVCodecContext *dec_ctx = avcodec_alloc_context3(avcodec_find_decoder(in_stream->codecpar->codec_id));
    avcodec_parameters_to_context(dec_ctx, in_stream->codecpar);
    avcodec_open2(dec_ctx, dec_ctx->codec, NULL);

    // Read, decode, encode and write frames
    int frame_index = 0;
    while (av_read_frame(input_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            ret = avcodec_send_packet(dec_ctx, pkt);
            if (ret < 0) continue;

            while (ret >= 0) {
                ret = avcodec_receive_frame(dec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) continue;

                frame->pts = frame_index++;

                ret = avcodec_send_frame(codec_ctx, frame);
                if (ret < 0) continue;

                while (ret >= 0) {
                    ret = avcodec_receive_packet(codec_ctx, pkt);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                    if (ret < 0) continue;

                    pkt->stream_index = 0;
                    av_packet_rescale_ts(pkt, codec_ctx->time_base, out_stream->time_base);
                    ret = av_interleaved_write_frame(output_ctx, pkt);
                    if (ret < 0) {
                        handle_error("Error writing frame");
                    }
                }
            }
        }
        av_packet_unref(pkt);
    }

    // Flush encoder
    avcodec_send_frame(codec_ctx, NULL);
    while (1) {
        ret = avcodec_receive_packet(codec_ctx, pkt);
        if (ret == AVERROR_EOF) break;
        if (ret < 0) continue;

        pkt->stream_index = 0;
        av_packet_rescale_ts(pkt, codec_ctx->time_base, out_stream->time_base);
        ret = av_interleaved_write_frame(output_ctx, pkt);
        if (ret < 0) {
            handle_error("Error writing frame");
        }
    }

    // Write trailer
    av_write_trailer(output_ctx);

    // Cleanup
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&codec_ctx);
    av_frame_free(&frame);
    av_packet_free(&pkt);
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
    
    encode(argv[1], argv[2]);
    return 0;
}