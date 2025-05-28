#include "filters/optical_flow.h"
#include "utils/cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <algorithm>

texture<float, cudaTextureType2D, cudaReadModeElementType> tex_prev_frame;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_curr_frame;

#define TILE_SIZE 32
#define HALF_WINDOW 7  

__global__ void rgb_to_grayscale_kernel(const unsigned char* rgb_image, float* gray_image, 
                                        int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        float gray = 0.0f;
        
        if (channels == 3) {
            gray = 0.299f * rgb_image[idx] + 0.587f * rgb_image[idx + 1] + 0.114f * rgb_image[idx + 2];
        } else {
            gray = rgb_image[idx];
        }
        
        gray_image[y * width + x] = gray / 255.0f;
    }
}

__global__ void compute_derivatives_kernel(float* Ix, float* Iy, float* It,
                                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1] / 8
        float dx = (-tex2D(tex_prev_frame, x-1, y-1) + tex2D(tex_prev_frame, x+1, y-1) +
                    -2.0f * tex2D(tex_prev_frame, x-1, y) + 2.0f * tex2D(tex_prev_frame, x+1, y) +
                    -tex2D(tex_prev_frame, x-1, y+1) + tex2D(tex_prev_frame, x+1, y+1)) / 8.0f;
        
        // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1] / 8
        float dy = (-tex2D(tex_prev_frame, x-1, y-1) - 2.0f * tex2D(tex_prev_frame, x, y-1) - tex2D(tex_prev_frame, x+1, y-1) +
                     tex2D(tex_prev_frame, x-1, y+1) + 2.0f * tex2D(tex_prev_frame, x, y+1) + tex2D(tex_prev_frame, x+1, y+1)) / 8.0f;
        
        // Temporal derivative
        float dt = tex2D(tex_curr_frame, x, y) - tex2D(tex_prev_frame, x, y);
        
        int idx = y * width + x;
        Ix[idx] = dx;
        Iy[idx] = dy;
        It[idx] = dt;
    }
}

// Lucas-Kanade optical flow kernel with shared memory optimization
__global__ void lucas_kanade_kernel(const float* Ix, const float* Iy, const float* It,
                                   FlowVector* flow, int width, int height, 
                                   int window_size, float min_eigenvalue) {
    // Shared memory for derivatives
    extern __shared__ float shared_mem[];
    float* s_Ix = shared_mem;
    float* s_Iy = s_Ix + (TILE_SIZE + 2 * HALF_WINDOW) * (TILE_SIZE + 2 * HALF_WINDOW);
    float* s_It = s_Iy + (TILE_SIZE + 2 * HALF_WINDOW) * (TILE_SIZE + 2 * HALF_WINDOW);
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tid_x;
    int y = blockIdx.y * TILE_SIZE + tid_y;
    
    // Load data into shared memory with halo regions
    for (int dy = tid_y - HALF_WINDOW; dy <= TILE_SIZE + HALF_WINDOW; dy += blockDim.y) {
        for (int dx = tid_x - HALF_WINDOW; dx <= TILE_SIZE + HALF_WINDOW; dx += blockDim.x) {
            int sx = dx + HALF_WINDOW;
            int sy = dy + HALF_WINDOW;
            int gx = blockIdx.x * TILE_SIZE + dx;
            int gy = blockIdx.y * TILE_SIZE + dy;
            
            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                int idx = gy * width + gx;
                s_Ix[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx] = Ix[idx];
                s_Iy[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx] = Iy[idx];
                s_It[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx] = It[idx];
            } else {
                s_Ix[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx] = 0.0f;
                s_Iy[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx] = 0.0f;
                s_It[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (x < width && y < height) {
        // Compute Lucas-Kanade matrix elements
        float A11 = 0.0f, A12 = 0.0f, A22 = 0.0f;
        float b1 = 0.0f, b2 = 0.0f;
        
        int half_win = window_size / 2;
        
        // Use shared memory for window operations
        for (int wy = -half_win; wy <= half_win; wy++) {
            for (int wx = -half_win; wx <= half_win; wx++) {
                int sx = tid_x + wx + HALF_WINDOW;
                int sy = tid_y + wy + HALF_WINDOW;
                
                float ix = s_Ix[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx];
                float iy = s_Iy[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx];
                float it = s_It[sy * (TILE_SIZE + 2 * HALF_WINDOW) + sx];
                
                A11 += ix * ix;
                A12 += ix * iy;
                A22 += iy * iy;
                b1 += -ix * it;
                b2 += -iy * it;
            }
        }
        
        // Solve 2x2 linear system using determinant
        float det = A11 * A22 - A12 * A12;
        
        // Check eigenvalue threshold (approximation using determinant and trace)
        float trace = A11 + A22;
        float min_eig = 0.5f * (trace - sqrtf(trace * trace - 4.0f * det));
        
        if (fabsf(det) > 1e-6f && min_eig > min_eigenvalue) {
            flow[y * width + x].u = (A22 * b1 - A12 * b2) / det;
            flow[y * width + x].v = (-A12 * b1 + A11 * b2) / det;
        } else {
            flow[y * width + x].u = 0.0f;
            flow[y * width + x].v = 0.0f;
        }
    }
}

// Pyramidal Lucas-Kanade implementation
void compute_optical_flow_gpu(const Image& prev_frame, const Image& curr_frame, 
                             FlowVector* flow_output, const OpticalFlowParams& params) {
    if (prev_frame.width != curr_frame.width || prev_frame.height != curr_frame.height) {
        fprintf(stderr, "Error: Frame dimensions must match\n");
        return;
    }
    
    int width = prev_frame.width;
    int height = prev_frame.height;
    
    // Convert to grayscale
    float *d_prev_gray, *d_curr_gray;
    size_t gray_size = width * height * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_prev_gray, gray_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_curr_gray, gray_size));
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    // Convert images to grayscale
    unsigned char *d_prev_rgb, *d_curr_rgb;
    size_t rgb_size = width * height * prev_frame.channels * sizeof(unsigned char);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_prev_rgb, rgb_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_curr_rgb, rgb_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_prev_rgb, prev_frame.data, rgb_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_curr_rgb, curr_frame.data, rgb_size, cudaMemcpyHostToDevice));
    
    rgb_to_grayscale_kernel<<<grid_size, block_size>>>(d_prev_rgb, d_prev_gray, 
                                                       width, height, prev_frame.channels);
    rgb_to_grayscale_kernel<<<grid_size, block_size>>>(d_curr_rgb, d_curr_gray, 
                                                       width, height, curr_frame.channels);
    
    // Set up texture memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *d_prev_array, *d_curr_array;
    
    CHECK_CUDA_ERROR(cudaMallocArray(&d_prev_array, &channelDesc, width, height));
    CHECK_CUDA_ERROR(cudaMallocArray(&d_curr_array, &channelDesc, width, height));
    
    CHECK_CUDA_ERROR(cudaMemcpyToArray(d_prev_array, 0, 0, d_prev_gray, gray_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToArray(d_curr_array, 0, 0, d_curr_gray, gray_size, cudaMemcpyDeviceToDevice));
    
    tex_prev_frame.addressMode[0] = cudaAddressModeClamp;
    tex_prev_frame.addressMode[1] = cudaAddressModeClamp;
    tex_prev_frame.filterMode = cudaFilterModeLinear;
    tex_prev_frame.normalized = false;
    
    tex_curr_frame.addressMode[0] = cudaAddressModeClamp;
    tex_curr_frame.addressMode[1] = cudaAddressModeClamp;
    tex_curr_frame.filterMode = cudaFilterModeLinear;
    tex_curr_frame.normalized = false;
    
    CHECK_CUDA_ERROR(cudaBindTextureToArray(tex_prev_frame, d_prev_array, channelDesc));
    CHECK_CUDA_ERROR(cudaBindTextureToArray(tex_curr_frame, d_curr_array, channelDesc));
    
    // Compute derivatives
    float *d_Ix, *d_Iy, *d_It;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Ix, gray_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Iy, gray_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_It, gray_size));
    
    compute_derivatives_kernel<<<grid_size, block_size>>>(d_Ix, d_Iy, d_It, width, height);
    
    // Allocate flow field
    FlowVector* d_flow;
    size_t flow_size = width * height * sizeof(FlowVector);
    CHECK_CUDA_ERROR(cudaMalloc(&d_flow, flow_size));
    CHECK_CUDA_ERROR(cudaMemset(d_flow, 0, flow_size));
    
    // Compute optical flow
    dim3 flow_block(16, 16);
    dim3 flow_grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    
    size_t shared_size = 3 * (TILE_SIZE + 2 * HALF_WINDOW) * (TILE_SIZE + 2 * HALF_WINDOW) * sizeof(float);
    lucas_kanade_kernel<<<flow_grid, flow_block, shared_size>>>(
        d_Ix, d_Iy, d_It, d_flow, width, height, params.window_size, params.min_eigenvalue);
    
    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(flow_output, d_flow, flow_size, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaUnbindTexture(tex_prev_frame));
    CHECK_CUDA_ERROR(cudaUnbindTexture(tex_curr_frame));
    CHECK_CUDA_ERROR(cudaFreeArray(d_prev_array));
    CHECK_CUDA_ERROR(cudaFreeArray(d_curr_array));
    CHECK_CUDA_ERROR(cudaFree(d_prev_rgb));
    CHECK_CUDA_ERROR(cudaFree(d_curr_rgb));
    CHECK_CUDA_ERROR(cudaFree(d_prev_gray));
    CHECK_CUDA_ERROR(cudaFree(d_curr_gray));
    CHECK_CUDA_ERROR(cudaFree(d_Ix));
    CHECK_CUDA_ERROR(cudaFree(d_Iy));
    CHECK_CUDA_ERROR(cudaFree(d_It));
    CHECK_CUDA_ERROR(cudaFree(d_flow));
}

// Flow visualization kernel - creates HSV color wheel representation
__global__ void flow_to_color_kernel(const FlowVector* flow, unsigned char* output, 
                                    int width, int height, float max_flow) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float u = flow[idx].u;
        float v = flow[idx].v;
        
        // Compute magnitude and angle
        float magnitude = sqrtf(u * u + v * v);
        float angle = atan2f(v, u);
        
        // Convert to HSV
        float h = (angle + M_PI) / (2.0f * M_PI); // Hue: angle mapped to [0,1]
        float s = fminf(magnitude / max_flow, 1.0f); // Saturation: normalized magnitude
        float val = 1.0f; // Value: always 1 for visibility
        
        // HSV to RGB conversion
        float c = val * s;
        float x_val = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
        float m = val - c;
        
        float r, g, b;
        if (h < 1.0f/6.0f) {
            r = c; g = x_val; b = 0;
        } else if (h < 2.0f/6.0f) {
            r = x_val; g = c; b = 0;
        } else if (h < 3.0f/6.0f) {
            r = 0; g = c; b = x_val;
        } else if (h < 4.0f/6.0f) {
            r = 0; g = x_val; b = c;
        } else if (h < 5.0f/6.0f) {
            r = x_val; g = 0; b = c;
        } else {
            r = c; g = 0; b = x_val;
        }
        
        output[idx * 3 + 0] = (unsigned char)((r + m) * 255.0f);
        output[idx * 3 + 1] = (unsigned char)((g + m) * 255.0f);
        output[idx * 3 + 2] = (unsigned char)((b + m) * 255.0f);
    }
}

void flow_to_color(const FlowVector* flow, int width, int height, Image& output) {
    output.width = width;
    output.height = height;
    output.channels = 3;
    output.data = new unsigned char[width * height * 3];
    
    // Upload flow to device
    FlowVector* d_flow;
    size_t flow_size = width * height * sizeof(FlowVector);
    CHECK_CUDA_ERROR(cudaMalloc(&d_flow, flow_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_flow, flow, flow_size, cudaMemcpyHostToDevice));
    
    // Allocate output
    unsigned char* d_output;
    size_t output_size = width * height * 3 * sizeof(unsigned char);
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    flow_to_color_kernel<<<grid_size, block_size>>>(d_flow, d_output, width, height, 10.0f);
    
    CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, output_size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_flow));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}