#ifndef BRIGHTNESS_CONTRAST_H
#define BRIGHTNESS_CONTRAST_H

#include "image_io.h"

void adjust_brightness_contrast(const Image& input, Image& output, float brightness, float contrast);

#endif // BRIGHTNESS_CONTRAST_H