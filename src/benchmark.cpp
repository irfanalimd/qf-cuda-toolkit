#include "benchmark.h"
#include "filters/gaussian_blur.h"
#include "filters/edge_detection.h"
#include "filters/brightness_contrast.h"
#include "pipeline.h"
#include "logger.h"
#include <iostream>
#include <iomanip>

// CPU implementations (only for benchmarking)
void cpu_gaussian_blur(const Image& input, Image& output, float sigma);
void cpu_sobel_edge_detection(const Image& input, Image& output);
void cpu_adjust_brightness_contrast(const Image& input, Image& output, float brightness, float contrast);

std::vector<BenchmarkResult> benchmark_pipeline(const Image& input) {
    std::vector<BenchmarkResult> results;
    Pipeline pipeline;
    Image output;
    Timer timer;

    // Benchmark Gaussian Blur
    {
        BenchmarkResult result;
        result.filter_name = "Gaussian Blur";

        pipeline.clear_filters();
        pipeline.add_filter(FilterType::GaussianBlur, 2.0f);

        timer.start();
        pipeline.process(input, output);
        timer.stop();
        result.gpu_time = timer.elapsed();

        timer.start();
        cpu_gaussian_blur(input, output, 2.0f);
        timer.stop();
        result.cpu_time = timer.elapsed();

        result.speedup = result.cpu_time / result.gpu_time;
        results.push_back(result);
    }

    // Benchmark Edge Detection
    {
        BenchmarkResult result;
        result.filter_name = "Edge Detection";

        pipeline.clear_filters();
        pipeline.add_filter(FilterType::EdgeDetection);

        timer.start();
        pipeline.process(input, output);
        timer.stop();
        result.gpu_time = timer.elapsed();

        timer.start();
        cpu_sobel_edge_detection(input, output);
        timer.stop();
        result.cpu_time = timer.elapsed();

        result.speedup = result.cpu_time / result.gpu_time;
        results.push_back(result);
    }

    // Benchmark Brightness/Contrast Adjustment
    {
        BenchmarkResult result;
        result.filter_name = "Brightness/Contrast";

        pipeline.clear_filters();
        pipeline.add_filter(FilterType::BrightnessContrast, 0.2f, 1.2f);

        timer.start();
        pipeline.process(input, output);
        timer.stop();
        result.gpu_time = timer.elapsed();

        timer.start();
        cpu_adjust_brightness_contrast(input, output, 0.2f, 1.2f);
        timer.stop();
        result.cpu_time = timer.elapsed();

        result.speedup = result.cpu_time / result.gpu_time;
        results.push_back(result);
    }

    return results;
}

void run_benchmarks(const std::string& input_filename) {
    Logger::log(LogLevel::INFO, "Starting benchmarks for: " + input_filename);

    Image input = load_image(input_filename.c_str());
    std::vector<BenchmarkResult> results = benchmark_pipeline(input);

    std::cout << std::setw(25) << "Filter" << std::setw(15) << "GPU Time (ms)" << std::setw(15) << "CPU Time (ms)" << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (const auto& result : results) {
        std::cout << std::setw(25) << result.filter_name
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.gpu_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.cpu_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.speedup << "x" << std::endl;

        Logger::log(LogLevel::INFO, "Benchmark - " + result.filter_name + ": GPU=" + std::to_string(result.gpu_time) + 
                    "ms, CPU=" + std::to_string(result.cpu_time) + "ms, Speedup=" + std::to_string(result.speedup) + "x");
    }

    // Calculate and display total times and overall speedup
    double total_gpu_time = 0.0, total_cpu_time = 0.0;
    for (const auto& result : results) {
        total_gpu_time += result.gpu_time;
        total_cpu_time += result.cpu_time;
    }
    double overall_speedup = total_cpu_time / total_gpu_time;

    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::setw(25) << "Total"
              << std::setw(15) << std::fixed << std::setprecision(2) << total_gpu_time
              << std::setw(15) << std::fixed << std::setprecision(2) << total_cpu_time
              << std::setw(15) << std::fixed << std::setprecision(2) << overall_speedup << "x" << std::endl;

    Logger::log(LogLevel::INFO, "Benchmark completed. Overall speedup: " + std::to_string(overall_speedup) + "x");

    // Clean up
    free_image(input);
}


// CPU implementations (only for benchmarking)
void cpu_gaussian_blur(const Image& input, Image& output, float sigma) {
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[input.width * input.height * input.channels];

    int kernel_size = static_cast<int>(ceil(sigma * 6)) | 1;
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

void cpu_sobel_edge_detection(const Image& input, Image& output) {
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[input.width * input.height * input.channels];

    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            for (int c = 0; c < input.channels; ++c) {
                float gx = 0.0f, gy = 0.0f;
                
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ix = std::min(std::max(x + kx, 0), input.width - 1);
                        int iy = std::min(std::max(y + ky, 0), input.height - 1);
                        float pixel = input.data[(iy * input.width + ix) * input.channels + c];
                        
                        gx += pixel * sobel_x[ky + 1][kx + 1];
                        gy += pixel * sobel_y[ky + 1][kx + 1];
                    }
                }

                float magnitude = std::sqrt(gx * gx + gy * gy);
                output.data[(y * input.width + x) * input.channels + c] = static_cast<unsigned char>(std::min(magnitude, 255.0f));
            }
        }
    }
}

void cpu_adjust_brightness_contrast(const Image& input, Image& output, float brightness, float contrast) {
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[input.width * input.height * input.channels];

    for (int i = 0; i < input.width * input.height * input.channels; ++i) {
        float pixel = input.data[i] / 255.0f;
        pixel = (pixel - 0.5f) * contrast + 0.5f + brightness;
        pixel = std::max(0.0f, std::min(pixel, 1.0f));
        output.data[i] = static_cast<unsigned char>(pixel * 255.0f);
    }
}


