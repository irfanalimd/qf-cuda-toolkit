#ifndef CPU_FILTERS_H
#define CPU_FILTERS_H

#include "image_io.h"

void cpu_gaussian_blur(const Image& input, Image& output, float sigma);

#endif // CPU_FILTERS_H