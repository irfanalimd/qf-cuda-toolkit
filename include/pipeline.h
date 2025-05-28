#ifndef PIPELINE_H
#define PIPELINE_H

#include "image_io.h"
#include <vector>
#include <functional>

enum class FilterType {
    GaussianBlur,
    EdgeDetection,
    BrightnessContrast,
    OpticalFlow  
};

struct FilterOperation {
    FilterType type;
    std::function<void(const Image&, Image&)> apply;
    float parameter1;
    float parameter2;  
};

struct OpticalFlowContext {
    Image* prev_frame = nullptr;
    bool initialized = false;
};

class Pipeline {
public:
    Pipeline() : flow_context(new OpticalFlowContext()) {}
    ~Pipeline() { delete flow_context; }
    
    void add_filter(FilterType type, float parameter1 = 0.0f, float parameter2 = 0.0f);
    void process(const Image& input, Image& output);
    void clear_filters() { filters.clear(); }
    

    void process_video_frame(const Image& input, Image& output);
    void reset_video_context();

private:
    std::vector<FilterOperation> filters;
    OpticalFlowContext* flow_context;
};

void process_image(const char* input_filename, const char* output_filename);
void process_video_with_flow(const char* input_pattern, const char* output_pattern, int start_frame, int end_frame);

#endif 