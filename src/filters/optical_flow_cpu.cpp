#include "filters/optical_flow.h"
#include <cmath>
#include <algorithm>
#include <vector>

// Helper function to convert RGB to grayscale
static void rgb_to_grayscale_cpu(const unsigned char* rgb, float* gray, 
                                int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            if (channels == 3) {
                gray[y * width + x] = (0.299f * rgb[idx] + 
                                       0.587f * rgb[idx + 1] + 
                                       0.114f * rgb[idx + 2]) / 255.0f;
            } else {
                gray[y * width + x] = rgb[idx] / 255.0f;
            }
        }
    }
}

// Compute image derivatives
static void compute_derivatives_cpu(const float* prev, const float* curr,
                                   float* Ix, float* Iy, float* It,
                                   int width, int height) {
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            
            // Sobel X derivative
            Ix[idx] = (-prev[(y-1) * width + (x-1)] + prev[(y-1) * width + (x+1)] +
                       -2.0f * prev[y * width + (x-1)] + 2.0f * prev[y * width + (x+1)] +
                       -prev[(y+1) * width + (x-1)] + prev[(y+1) * width + (x+1)]) / 8.0f;
            
            // Sobel Y derivative
            Iy[idx] = (-prev[(y-1) * width + (x-1)] - 2.0f * prev[(y-1) * width + x] - prev[(y-1) * width + (x+1)] +
                        prev[(y+1) * width + (x-1)] + 2.0f * prev[(y+1) * width + x] + prev[(y+1) * width + (x+1)]) / 8.0f;
            
            // Temporal derivative
            It[idx] = curr[idx] - prev[idx];
        }
    }
}

void compute_optical_flow_cpu(const Image& prev_frame, const Image& curr_frame, 
                             FlowVector* flow_output, const OpticalFlowParams& params) {
    if (prev_frame.width != curr_frame.width || prev_frame.height != curr_frame.height) {
        fprintf(stderr, "Error: Frame dimensions must match\n");
        return;
    }
    
    int width = prev_frame.width;
    int height = prev_frame.height;
    
    // Convert to grayscale
    std::vector<float> prev_gray(width * height);
    std::vector<float> curr_gray(width * height);
    
    rgb_to_grayscale_cpu(prev_frame.data, prev_gray.data(), width, height, prev_frame.channels);
    rgb_to_grayscale_cpu(curr_frame.data, curr_gray.data(), width, height, curr_frame.channels);
    
    // Compute derivatives
    std::vector<float> Ix(width * height, 0.0f);
    std::vector<float> Iy(width * height, 0.0f);
    std::vector<float> It(width * height, 0.0f);
    
    compute_derivatives_cpu(prev_gray.data(), curr_gray.data(), 
                           Ix.data(), Iy.data(), It.data(), width, height);
    
    // Lucas-Kanade optical flow computation
    int half_window = params.window_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float A11 = 0.0f, A12 = 0.0f, A22 = 0.0f;
            float b1 = 0.0f, b2 = 0.0f;
            
            // Build linear system over window
            for (int wy = -half_window; wy <= half_window; ++wy) {
                for (int wx = -half_window; wx <= half_window; ++wx) {
                    int nx = std::min(std::max(x + wx, 0), width - 1);
                    int ny = std::min(std::max(y + wy, 0), height - 1);
                    int idx = ny * width + nx;
                    
                    float ix = Ix[idx];
                    float iy = Iy[idx];
                    float it = It[idx];
                    
                    A11 += ix * ix;
                    A12 += ix * iy;
                    A22 += iy * iy;
                    b1 += -ix * it;
                    b2 += -iy * it;
                }
            }
            
            // Solve 2x2 system
            float det = A11 * A22 - A12 * A12;
            float trace = A11 + A22;
            float min_eig = 0.5f * (trace - std::sqrt(trace * trace - 4.0f * det));
            
            int idx = y * width + x;
            if (std::abs(det) > 1e-6f && min_eig > params.min_eigenvalue) {
                flow_output[idx].u = (A22 * b1 - A12 * b2) / det;
                flow_output[idx].v = (-A12 * b1 + A11 * b2) / det;
            } else {
                flow_output[idx].u = 0.0f;
                flow_output[idx].v = 0.0f;
            }
        }
    }
}

// Simple arrow visualization for optical flow
void visualize_flow_field(const FlowVector* flow, int width, int height, Image& output, float scale) {
    // Copy input or create blank image
    output.width = width;
    output.height = height;
    output.channels = 3;
    output.data = new unsigned char[width * height * 3];
    
    // Initialize with dark background
    std::fill(output.data, output.data + width * height * 3, 50);
    
    // Draw flow vectors as arrows every N pixels
    int step = 10;
    for (int y = step/2; y < height; y += step) {
        for (int x = step/2; x < width; x += step) {
            int idx = y * width + x;
            float u = flow[idx].u * scale;
            float v = flow[idx].v * scale;
            
            // Draw arrow from (x,y) to (x+u, y+v)
            int x2 = std::min(std::max(int(x + u), 0), width - 1);
            int y2 = std::min(std::max(int(y + v), 0), height - 1);
            
            // Simple line drawing (Bresenham's algorithm would be better)
            float steps = std::max(std::abs(u), std::abs(v));
            if (steps > 0) {
                for (int i = 0; i <= steps; ++i) {
                    int px = x + (x2 - x) * i / steps;
                    int py = y + (y2 - y) * i / steps;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        output.data[(py * width + px) * 3 + 0] = 255; // Red channel
                        output.data[(py * width + px) * 3 + 1] = 100;
                        output.data[(py * width + px) * 3 + 2] = 100;
                    }
                }
            }
        }
    }
}

// Warp image using flow field
void warp_image_with_flow(const Image& input, const FlowVector* flow, Image& output) {
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[input.width * input.height * input.channels];
    
    // Initialize output to black
    std::fill(output.data, output.data + input.width * input.height * input.channels, 0);
    
    // Backward warping with bilinear interpolation
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            int idx = y * input.width + x;
            
            // Source coordinates
            float src_x = x - flow[idx].u;
            float src_y = y - flow[idx].v;
            
            // Bilinear interpolation
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            if (x0 >= 0 && x1 < input.width && y0 >= 0 && y1 < input.height) {
                float dx = src_x - x0;
                float dy = src_y - y0;
                
                for (int c = 0; c < input.channels; ++c) {
                    float v00 = input.data[(y0 * input.width + x0) * input.channels + c];
                    float v10 = input.data[(y0 * input.width + x1) * input.channels + c];
                    float v01 = input.data[(y1 * input.width + x0) * input.channels + c];
                    float v11 = input.data[(y1 * input.width + x1) * input.channels + c];
                    
                    float v0 = v00 * (1 - dx) + v10 * dx;
                    float v1 = v01 * (1 - dx) + v11 * dx;
                    float v = v0 * (1 - dy) + v1 * dy;
                    
                    output.data[(y * input.width + x) * input.channels + c] = (unsigned char)v;
                }
            }
        }
    }
}