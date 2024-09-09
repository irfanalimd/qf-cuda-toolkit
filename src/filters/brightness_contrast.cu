#include "filters/brightness_contrast.h"
#include "utils/cuda_utils.h"
#include <cuda_runtime.h>

__global__ void brightness_contrast_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels, float brightness, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            int idx = (y * width + x) * channels + c;
            float pixel = input[idx] / 255.0f;
            pixel = (pixel - 0.5f) * contrast + 0.5f + brightness;
            pixel = max(0.0f, min(pixel, 1.0f));
            output[idx] = static_cast<unsigned char>(pixel * 255.0f);
        }
    }
}

void adjust_brightness_contrast(const Image& input, Image& output, float brightness, float contrast) {
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;

    size_t image_size = input.width * input.height * input.channels * sizeof(unsigned char);
    unsigned char* d_input;
    unsigned char* d_output;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, image_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, image_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, image_size, cudaMemcpyHostToDevice));

    dim3 block_size(16, 16);
    dim3 grid_size((input.width + block_size.x - 1) / block_size.x, (input.height + block_size.y - 1) / block_size.y);

    brightness_contrast_kernel<<<grid_size, block_size>>>(d_input, d_output, input.width, input.height, input.channels, brightness, contrast);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    output.data = new unsigned char[image_size];
    CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, image_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}