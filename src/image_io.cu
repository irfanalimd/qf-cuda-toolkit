#include "image_io.h"
#include "utils/cuda_utils.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image load_image(const char* filename) {
    Image img;
    img.data = stbi_load(filename, &img.width, &img.height, &img.channels, 0);
    if (img.data == nullptr) {
        fprintf(stderr, "Error loading image %s\n", filename);
        exit(1);
    }
    return img;
}

void save_image(const char* filename, const Image& img) {
    stbi_write_png(filename, img.width, img.height, img.channels, img.data, img.width * img.channels);
}

void free_image(Image& img) {
    stbi_image_free(img.data);
    img.data = nullptr;
}