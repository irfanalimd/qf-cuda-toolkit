#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include "image_io.h"
#include <vector>

struct FlowVector {
    float u; 
    float v;
};

struct OpticalFlowParams {
    int window_size = 15;        
    int pyramid_levels = 3;     
    float pyramid_scale = 0.5f;  
    int iterations = 10;         
    float min_eigenvalue = 0.001f; 
};

// Main optical flow computation function
void compute_optical_flow_gpu(const Image& prev_frame, const Image& curr_frame, 
                             FlowVector* flow_output, const OpticalFlowParams& params = OpticalFlowParams());

// CPU implementation for benchmarking
void compute_optical_flow_cpu(const Image& prev_frame, const Image& curr_frame, 
                             FlowVector* flow_output, const OpticalFlowParams& params = OpticalFlowParams());

// Visualization functions
void visualize_flow_field(const FlowVector* flow, int width, int height, Image& output, float scale = 1.0f);
void flow_to_color(const FlowVector* flow, int width, int height, Image& output);

// Utility functions
void warp_image_with_flow(const Image& input, const FlowVector* flow, Image& output);

#endif 