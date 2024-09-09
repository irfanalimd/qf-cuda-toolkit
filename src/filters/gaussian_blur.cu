#include "filters/gaussian_blur.h"
#include "utils/cuda_utils.h"
#include <cmath>

__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels, float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = min(max(x + kx - kernel_size / 2, 0), width - 1);
                    int iy = min(max(y + ky - kernel_size / 2, 0), height - 1);
                    sum += input[(iy * width + ix) * channels + c] * kernel[ky * kernel_size + kx];
                }
            }
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum);
        }
    }
}

void gaussian_blur(const Image& input, Image& output, float sigma) {
    const int kernel_size = static_cast<int>(ceil(sigma * 6)) | 1; // Ensure odd size
    float* h_kernel = new float[kernel_size * kernel_size];
    float sum = 0.0f;
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            int dx = x - kernel_size / 2;
            int dy = y - kernel_size / 2;
            float value = exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            h_kernel[y * kernel_size + x] = value;
            sum += value;
        }
    }
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        h_kernel[i] /= sum;
    }

    float* d_kernel;
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    unsigned char* d_input;
    unsigned char* d_output;
    size_t image_size = input.width * input.height * input.channels * sizeof(unsigned char);
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, image_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, image_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, image_size, cudaMemcpyHostToDevice));

    dim3 block_size(16, 16);
    dim3 grid_size((input.width + block_size.x - 1) / block_size.x, (input.height + block_size.y - 1) / block_size.y);

    gaussian_blur_kernel<<<grid_size, block_size>>>(d_input, d_output, input.width, input.height, input.channels, d_kernel, kernel_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[image_size];
    CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, image_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
    delete[] h_kernel;
}

