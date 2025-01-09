<<<<<<< HEAD
# video-codec
=======
# README.md

# Video Codec Project

This project implements a video encoding and decoding system. The encoder compresses video files into a more compact format, while the decoder reconstructs the original video from the encoded file.

## Project Structure

```
video-codec
├── src
│   ├── encoder.c         # Implementation of video encoding functionality
│   ├── decoder.c         # Implementation of video decoding functionality
│   ├── codec_utils.c     # Utility functions for encoder and decoder
│   └── codec_utils.h     # Header file for utility functions
├── tests
│   ├── test_encoder.c     # Unit tests for the encoder
│   └── test_decoder.c     # Unit tests for the decoder
├── CMakeLists.txt        # Build configuration for CMake
├── Makefile               # Build automation using make
└── README.md              # Project documentation
```

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

## Dependencies

- FFmpeg libraries (libavcodec, libavformat, libswscale)

Make sure to install the necessary dependencies before building the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
>>>>>>> c227d51 (initial)
