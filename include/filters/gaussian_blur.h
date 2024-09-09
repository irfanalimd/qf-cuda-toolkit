#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include "image_io.h"

void gaussian_blur(const Image& input, Image& output, float sigma);

#endif // GAUSSIAN_BLUR_H