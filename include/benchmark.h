#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "image_io.h"
#include <string>
#include <chrono>
#include <vector>

class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    void stop() { end_time = std::chrono::high_resolution_clock::now(); }
    double elapsed() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to milliseconds
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
};

struct BenchmarkResult {
    std::string filter_name;
    double gpu_time;
    double cpu_time;
    double speedup;
};

void run_benchmarks(const std::string& input_filename);
std::vector<BenchmarkResult> benchmark_pipeline(const Image& input);

#endif // BENCHMARK_H