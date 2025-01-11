#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
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
    unsigned char blue;
    unsigned char green;
    unsigned char red;
} Pixel;

// CUDA kernel to invert colors
__global__ void invert_colors_kernel(Pixel* d_pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        d_pixels[idx].red = 255 - d_pixels[idx].red;
        d_pixels[idx].green = 255 - d_pixels[idx].green;
        d_pixels[idx].blue = 255 - d_pixels[idx].blue;
    }
}

// (Other CUDA kernels: encode_bmp_kernel and decode_bmp_kernel are unchanged)

// CUDA kernel to  BMP image into raw pixel data
__global__ void encode_bmp_kernel(Pixel* d_pixels, unsigned char* d_raw_data, int width, int height, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelIdx = y * width + x;
        int rawIdx = y * (width * 3 + padding) + x * 3;

        d_raw_data[rawIdx] = d_pixels[pixelIdx].blue;
        d_raw_data[rawIdx + 1] = d_pixels[pixelIdx].green;
        d_raw_data[rawIdx + 2] = d_pixels[pixelIdx].red;
    }
}

// CUDA kernel to decode raw BMP pixel data
__global__ void decode_bmp_kernel(unsigned char* d_raw_data, Pixel* d_pixels, int width, int height, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rawIdx = y * (width * 3 + padding) + x * 3;
        int pixelIdx = y * width + x;

        d_pixels[pixelIdx].blue = d_raw_data[rawIdx];
        d_pixels[pixelIdx].green = d_raw_data[rawIdx + 1];
        d_pixels[pixelIdx].red = d_raw_data[rawIdx + 2];
    }
}

// Function to read BMP file and decode pixel data
Pixel* read_bmp(const char* filename, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return NULL;
    }

    BMPHeader header;
    BMPInfoHeader info;

    // Read headers
    fread(&header, sizeof(BMPHeader), 1, file);
    fread(&info, sizeof(BMPInfoHeader), 1, file);

    // Verify that it's a BMP file
    if (header.type != 0x4D42) {
        fprintf(stderr, "Not a BMP file\n");
        fclose(file);
        return NULL;
    }

    // Verify color depth
    if (info.bits != 24) {
        fprintf(stderr, "Only 24-bit BMP files are supported\n");
        fclose(file);
        return NULL;
    }

    *width = info.width;
    *height = info.height;

    int padding = (4 - (info.width * 3) % 4) % 4;
    int rawSize = (*height) * ((*width) * 3 + padding);

    // Allocate memory for raw data and pixels
    unsigned char* h_raw_data = (unsigned char*)malloc(rawSize);
    Pixel* h_pixels = (Pixel*)malloc((*width) * (*height) * sizeof(Pixel));

    // Read raw pixel data
    fseek(file, header.offset, SEEK_SET);
    fread(h_raw_data, rawSize, 1, file);
    fclose(file);

    // Allocate GPU memory
    unsigned char* d_raw_data;
    Pixel* d_pixels;
    cudaMalloc((void**)&d_raw_data, rawSize);
    cudaMalloc((void**)&d_pixels, (*width) * (*height) * sizeof(Pixel));

    // Copy raw data to GPU
    cudaMemcpy(d_raw_data, h_raw_data, rawSize, cudaMemcpyHostToDevice);

    // Launch kernel to decode BMP
    dim3 blockSize(16, 16);
    dim3 gridSize((*width + blockSize.x - 1) / blockSize.x, 
                  (*height + blockSize.y - 1) / blockSize.y);
    decode_bmp_kernel<<<gridSize, blockSize>>>(d_raw_data, d_pixels, *width, *height, padding);
    cudaDeviceSynchronize();

    // Copy decoded pixels back to CPU
    cudaMemcpy(h_pixels, d_pixels, (*width) * (*height) * sizeof(Pixel), cudaMemcpyDeviceToHost);

    // Free GPU and raw memory
    cudaFree(d_raw_data);
    cudaFree(d_pixels);
    free(h_raw_data);

    return h_pixels;
}

// Function to save BMP file by encoding pixel data
void save_bmp(const char* filename, Pixel* pixels, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Could not create file: %s\n", filename);
        return;
    }

    int padding = (4 - (width * 3) % 4) % 4;
    int rawSize = height * (width * 3 + padding);

    // Allocate memory for raw data
    unsigned char* h_raw_data = (unsigned char*)malloc(rawSize);

    // Allocate GPU memory
    unsigned char* d_raw_data;
    Pixel* d_pixels;
    cudaMalloc((void**)&d_raw_data, rawSize);
    cudaMalloc((void**)&d_pixels, width * height * sizeof(Pixel));

    // Copy pixel data to GPU
    cudaMemcpy(d_pixels, pixels, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Launch kernel to encode BMP
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    encode_bmp_kernel<<<gridSize, blockSize>>>(d_pixels, d_raw_data, width, height, padding);
    cudaDeviceSynchronize();

    // Copy encoded raw data back to CPU
    cudaMemcpy(h_raw_data, d_raw_data, rawSize, cudaMemcpyDeviceToHost);

    // Prepare headers
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

    // Write headers
    fwrite(&header, sizeof(BMPHeader), 1, file);
    fwrite(&info, sizeof(BMPInfoHeader), 1, file);

    // Write raw pixel data
    fwrite(h_raw_data, rawSize, 1, file);

    fclose(file);

    // Free GPU and raw memory
    cudaFree(d_raw_data);
    cudaFree(d_pixels);
    free(h_raw_data);
}


void process_images_in_directory(const char* input_dir, const char* output_dir) {
    DIR* dir = opendir(input_dir);
    if (!dir) {
        fprintf(stderr, "Could not open directory: %s\n", input_dir);
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {  // Only regular files
            const char* file_name = entry->d_name;
            const char* ext = strrchr(file_name, '.');

            if (!ext || strcmp(ext, ".bmp") != 0) {
                continue;  // Skip non-BMP files
            }

            char input_path[512], output_path[512];
            snprintf(input_path, sizeof(input_path), "%s/%s", input_dir, file_name);
            snprintf(output_path, sizeof(output_path), "%s/%s", output_dir, file_name);

            int width, height;
            Pixel* pixels = read_bmp(input_path, &width, &height);
            if (!pixels) {
                fprintf(stderr, "Failed to process file: %s\n", input_path);
                continue;
            }

            int num_pixels = width * height;

            // Allocate memory on GPU
            Pixel* d_pixels;
            cudaMalloc((void**)&d_pixels, num_pixels * sizeof(Pixel));

            // Copy pixel data to GPU
            cudaMemcpy(d_pixels, pixels, num_pixels * sizeof(Pixel), cudaMemcpyHostToDevice);

            // Define block and grid sizes
            dim3 blockSize(16, 16);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                          (height + blockSize.y - 1) / blockSize.y);

            // Launch the kernel
            invert_colors_kernel<<<gridSize, blockSize>>>(d_pixels, width, height);
            cudaDeviceSynchronize();

            // Copy results back to CPU
            cudaMemcpy(pixels, d_pixels, num_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost);

            // Save the processed image
            save_bmp(output_path, pixels, width, height);

            // Free memory
            cudaFree(d_pixels);
            free(pixels);
        }
    }

    closedir(dir);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_video.mp4> <output_video.mp4>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    const char* input_dir = "tmp_frames";
    const char* output_dir = "tmp_processed_frames";

    struct stat st0 = {0};
    if (stat(input_dir, &st0) == -1) {
        mkdir(input_dir, 0700);
    }
    
    //Turn video into frames
    char command[1024];
    snprintf(command, sizeof(command), "ffmpeg -i %s %s/frame_%%04d.bmp", input_path, input_dir);
    system(command);

    // Create the output directory if it doesn't exist
    struct stat st1 = {0};
    if (stat(output_dir, &st1) == -1) {
        mkdir(output_dir, 0700);
    }

    // Process all BMP files in the input directory
    process_images_in_directory(input_dir, output_dir);

    // Combine processed frames into a video
    snprintf(command, sizeof(command), "ffmpeg -framerate 30 -i %s/frame_%%04d.bmp %s", output_dir, output_path);
    system(command);

    // Clean up
    snprintf(command, sizeof(command), "rm -rf %s %s", input_dir, output_dir);
    system(command);

    return 0;
}
