#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t type;              // Magic identifier: 0x4d42
    uint32_t size;              // File size in bytes
    uint16_t reserved1;         // Not used
    uint16_t reserved2;         // Not used
    uint32_t offset;            // Offset to image data in bytes
} BMPHeader;

typedef struct {
    uint32_t size;              // Header size in bytes
    int32_t width;              // Width of the image
    int32_t height;             // Height of the image
    uint16_t planes;            // Number of color planes
    uint16_t bits;              // Bits per pixel
    uint32_t compression;       // Compression type
    uint32_t imagesize;         // Image size in bytes
    int32_t xresolution;        // Pixels per meter
    int32_t yresolution;        // Pixels per meter
    uint32_t ncolors;           // Number of colors
    uint32_t importantcolors;   // Important colors
} BMPInfoHeader;
#pragma pack(pop)

typedef struct {
    unsigned char r, g, b;
} Pixel;

__global__ void process_frame_kernel(unsigned char** d_raw_data, int width, int height, int padding, int index) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rawIdx = y * (width * 3 + padding) + x * 3;
        int pixelIdx = y * width + x;

        d_raw_data[index][rawIdx] = 255 - d_raw_data[index][rawIdx];
        d_raw_data[index][rawIdx + 1] = 255 - d_raw_data[index][rawIdx + 1];
        d_raw_data[index][rawIdx + 2] = 255 - d_raw_data[index][rawIdx + 2];
    }
}

// Function to read BMP file and decode pixel data
void read_bmp(const char* filename, unsigned char** raw_data, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        *raw_data = NULL;
        return;
    }

    BMPHeader header;
    BMPInfoHeader info;

    // Read BMP header and info
    fread(&header, sizeof(BMPHeader), 1, file);
    fread(&info, sizeof(BMPInfoHeader), 1, file);

    // Verify that it's a BMP file
    if (header.type != 0x4D42) {
        fprintf(stderr, "Not a BMP file\n");
        fclose(file);
        return;
    }

    // Verify color depth
    if (info.bits != 24) {
        fprintf(stderr, "Only 24-bit BMP files are supported\n");
        fclose(file);
        *raw_data = NULL;
        return;
    }

    *width = info.width;
    *height = info.height;

    int padding = (4 - (info.width * 3) % 4) % 4;
    int rawSize = (*height) * ((*width) * 3 + padding);

    // Allocate memory for raw data
    *raw_data = (unsigned char*)malloc(rawSize);

    // Read raw pixel data
    fseek(file, header.offset, SEEK_SET);
    fread(*raw_data, rawSize, 1, file);
    fclose(file);
}

void save_bmp(const char* filename, unsigned char* raw_data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Could not create file: %s\n", filename);
        return;
    }

    int padding = (4 - (width * 3) % 4) % 4;
    int rawSize = height * (width * 3 + padding);

    BMPHeader header = {
        .type = 0x4D42,
        .size = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + rawSize,
        .reserved1 = 0,
        .reserved2 = 0,
        .offset = sizeof(BMPHeader) + sizeof(BMPInfoHeader)
    };

    BMPInfoHeader info = {
        .size = sizeof(BMPInfoHeader),
        .width = width,
        .height = height,
        .planes = 1,
        .bits = 24,
        .compression = 0,
        .imagesize = rawSize,
        .xresolution = 2835,  // 72 DPI
        .yresolution = 2835,  // 72 DPI
        .ncolors = 0,
        .importantcolors = 0
    };


    // Write BMP header and info
    fwrite(&header, sizeof(BMPHeader), 1, file);
    fwrite(&info, sizeof(BMPInfoHeader), 1, file);

    // Write raw pixel data
    fwrite(raw_data, rawSize, 1, file);
    fclose(file);
    free(raw_data);
}

void run_ffmpeg_command(const char* command) {
    int ret = system(command);
    if (ret != 0) {
        fprintf(stderr, "Error running command: %s\n", command);
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.mp4> <output.mp4>\n", argv[0]);
        return 1;
    }

    const char* input_video = argv[1];
    const char* output_video = argv[2];

    // Create frames directory
    mkdir("frames", 0777);
    mkdir("output", 0777);

    // Extract frames from video
    char ffmpeg_command[256];
    snprintf(ffmpeg_command, sizeof(ffmpeg_command), "ffmpeg -i %s frames/frame_%%04d.bmp", input_video);
    run_ffmpeg_command(ffmpeg_command);

    auto start = std::chrono::high_resolution_clock::now();

    // Process frames
    DIR* dir;
    struct dirent* entry;
    if (!(dir = opendir("frames"))) {
        fprintf(stderr, "Could not open frames directory\n");
        return 1;
    }

    int num_frames = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".bmp") != NULL) {
            num_frames++;
        }
    }
    closedir(dir);

    unsigned char** raw_data = (unsigned char**)malloc(num_frames * sizeof(unsigned char*));
    int* widths = (int*)malloc(num_frames * sizeof(int));
    int* heights = (int*)malloc(num_frames * sizeof(int));

    for (int i = 0; i < num_frames; ++i) {
        char frame_filename[256];
        snprintf(frame_filename, sizeof(frame_filename), "frames/frame_%04d.bmp", i + 1);
        read_bmp(frame_filename, &raw_data[i], &widths[i], &heights[i]);
        // printf("Frame %d: %d x %d\n", i, widths[i], heights[i]);
        if (raw_data[i] == NULL) {
            fprintf(stderr, "Failed to read BMP file: %s\n", frame_filename);
            return 1;
        }
    }

    unsigned char** d_raw_data_array;
    cudaMallocHost((void***)&d_raw_data_array, num_frames * sizeof(unsigned char*));

    // It is assumed that all frames have the same dimensions
    int padding = ((4 - (widths[0] * 3) % 4) % 4);
    int rawSize = heights[0] * (widths[0] * 3 + padding);

    for (int i = 0; i < num_frames; ++i) {
        cudaMalloc((void**)&d_raw_data_array[i], rawSize);
    }

    int num_streams = 8;
    cudaStream_t streams[num_streams];

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    
    dim3 blockSize(16, 16);
    dim3 gridSize((widths[0] + blockSize.x - 1) / blockSize.x, 
                    (heights[0] + blockSize.y - 1) / blockSize.y);
    for (int i = 0; i < num_frames; ++i) {
        cudaMemcpyAsync(d_raw_data_array[i], raw_data[i], rawSize, cudaMemcpyHostToDevice, streams[i % num_streams]);
        process_frame_kernel<<<gridSize, blockSize, 0, streams[i % num_streams]>>>(d_raw_data_array, widths[i], heights[i], padding, i);
        cudaMemcpyAsync(raw_data[i], d_raw_data_array[i], widths[i] * heights[i] * sizeof(Pixel), cudaMemcpyDeviceToHost, streams[i % num_streams]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < num_frames; ++i) {
        cudaFree(d_raw_data_array[i]);
    }

    cudaFree(d_raw_data_array);

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    for (int i = 0; i < num_frames; ++i) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "output/frame_%04d.bmp", i + 1);
        save_bmp(output_filename, raw_data[i], widths[i], heights[i]);
    }

    free(raw_data);
    free(widths);
    free(heights);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Total runtime: %f seconds\n", duration.count());

    // Combine frames into video
    snprintf(ffmpeg_command, sizeof(ffmpeg_command), "ffmpeg -framerate 30 -i output/frame_%%04d.bmp %s", output_video);
    run_ffmpeg_command(ffmpeg_command);
    
    // Remove frames directory
    snprintf(ffmpeg_command, sizeof(ffmpeg_command), "rm -rf frames");
    run_ffmpeg_command(ffmpeg_command);

    // Remove output directory
    snprintf(ffmpeg_command, sizeof(ffmpeg_command), "rm -rf output");
    run_ffmpeg_command(ffmpeg_command);
    


    return 0;
}