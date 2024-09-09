#include "filters/cpu_filters.h"
#include <cmath>
#include <vector>
#include <algorithm>

void cpu_gaussian_blur(const Image& input, Image& output, float sigma) {
    const int kernel_size = static_cast<int>(ceil(sigma * 6)) | 1; // Ensure odd size
    std::vector<float> kernel(kernel_size * kernel_size);
    
    float sum = 0.0f;
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            int dx = x - kernel_size / 2;
            int dy = y - kernel_size / 2;
            float value = exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            kernel[y * kernel_size + x] = value;
            sum += value;
        }
    }
    for (float& k : kernel) {
        k /= sum;
    }

    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[input.width * input.height * input.channels];

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            for (int c = 0; c < input.channels; ++c) {
                float sum = 0.0f;
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int ix = std::min(std::max(x + kx - kernel_size / 2, 0), input.width - 1);
                        int iy = std::min(std::max(y + ky - kernel_size / 2, 0), input.height - 1);
                        sum += input.data[(iy * input.width + ix) * input.channels + c] * kernel[ky * kernel_size + kx];
                    }
                }
                output.data[(y * input.width + x) * input.channels + c] = static_cast<unsigned char>(sum);
            }
        }
    }
}
