# Video Codec Project

This project implements a video encoding and decoding system. The encoder compresses video files into a more compact format, while the decoder reconstructs the original video from the encoded file.

## Dependencies

- FFmpeg libraries (libavcodec, libavformat, libswscale)

sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev

## Building the Project

To build the project, you can use either CMake or Make.

### Using CMake

1. Create a build directory:
   ```
   mkdir build
   cd build
   ```

2. Run CMake to configure the project:
   ```
   cmake ..
   ```

3. Build the project:
   ```
   make
   ```

### Using Make

Simply run:
```
make
```

## Running the Encoder and Decoder

After building the project, you can run the encoder and decoder from the command line. The usage is as follows:

### Encoder

```
./encoder <input_video_file> <output_encoded_file>
```

### Decoder

```
./decoder <input_encoded_file> <output_video_file>
```

