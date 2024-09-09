#ifndef PIPELINE_H
#define PIPELINE_H

#include "image_io.h"
#include <vector>
#include <functional>

enum class FilterType {
    GaussianBlur,
    EdgeDetection,
    BrightnessContrast  
};

struct FilterOperation {
    FilterType type;
    std::function<void(const Image&, Image&)> apply;
    float parameter1;
    float parameter2;  
};

class Pipeline {
public:
    void add_filter(FilterType type, float parameter1 = 0.0f, float parameter2 = 0.0f);
    void process(const Image& input, Image& output);
    void clear_filters() { filters.clear(); }

private:
    std::vector<FilterOperation> filters;
};

void process_image(const char* input_filename, const char* output_filename);

#endif // PIPELINE_H