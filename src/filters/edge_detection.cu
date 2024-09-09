#include "filters/edge_detection.h"
#include "utils/cuda_utils.h"
#include <cuda_runtime.h>

__global__ void sobel_edge_detection_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            float gx = 0.0f, gy = 0.0f;
            
            // Sobel operator
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = min(max(x + dx, 0), width - 1);
                    int ny = min(max(y + dy, 0), height - 1);
                    float pixel = input[(ny * width + nx) * channels + c];
                    
                    gx += pixel * (dx * (2 - abs(dy)));
                    gy += pixel * (dy * (2 - abs(dx)));
                }
            }

            float magnitude = sqrtf(gx * gx + gy * gy);
            output[(y * width + x) * channels + c] = (unsigned char)min(magnitude, 255.0f);
        }
    }
}

void sobel_edge_detection(const Image& input, Image& output) {
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

    sobel_edge_detection_kernel<<<grid_size, block_size>>>(d_input, d_output, input.width, input.height, input.channels);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    output.data = new unsigned char[image_size];
    CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, image_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}