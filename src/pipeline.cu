#include "pipeline.h"
#include "filters/gaussian_blur.h"
#include "filters/edge_detection.h"
#include "filters/brightness_contrast.h"
#include <iostream>

void Pipeline::add_filter(FilterType type, float parameter1, float parameter2) {
    FilterOperation op;
    op.type = type;
    op.parameter1 = parameter1;
    op.parameter2 = parameter2;

    switch (type) {
        case FilterType::GaussianBlur:
            op.apply = [parameter1](const Image& input, Image& output) {
                gaussian_blur(input, output, parameter1);
            };
            break;
        case FilterType::EdgeDetection:
            op.apply = [](const Image& input, Image& output) {
                sobel_edge_detection(input, output);
            };
            break;
        case FilterType::BrightnessContrast:
            op.apply = [parameter1, parameter2](const Image& input, Image& output) {
                adjust_brightness_contrast(input, output, parameter1, parameter2);
            };
            break;
    }

    filters.push_back(op);
}

void Pipeline::process(const Image& input, Image& output) {
    Image temp1 = input;
    Image temp2;

    for (size_t i = 0; i < filters.size(); ++i) {
        filters[i].apply(temp1, temp2);
        std::swap(temp1, temp2);
    }

    output = temp1;
}

void process_image(const char* input_filename, const char* output_filename) {
    Image input = load_image(input_filename);
    Image output;

    Pipeline pipeline;
    pipeline.add_filter(FilterType::GaussianBlur, 2.0f);
    pipeline.add_filter(FilterType::EdgeDetection);

    pipeline.process(input, output);

    save_image(output_filename, output);
    free_image(input);
    free_image(output);
}