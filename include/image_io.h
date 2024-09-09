#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cuda_runtime.h>

struct Image {
    int width;
    int height;
    int channels;
    unsigned char* data;
};

Image load_image(const char* filename);
void save_image(const char* filename, const Image& img);
void free_image(Image& img);

#endif // IMAGE_IO_H