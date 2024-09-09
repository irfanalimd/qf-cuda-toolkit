
#include "pipeline.h"
#include "benchmark.h"
#include "logger.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    Logger::init("image_processor.log");
    Logger::log(LogLevel::INFO, "Application started");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> [arguments]" << std::endl;
        std::cerr << "Modes: process, benchmark, gui" << std::endl;
        Logger::log(LogLevel::ERROR, "Invalid command line arguments");
        return 1;
    }

    std::string mode = argv[1];

    try {
        if (mode == "process") {
            if (argc != 4) {
                std::cerr << "Process mode usage: " << argv[0] << " process <input_image> <output_image>" << std::endl;
                Logger::log(LogLevel::ERROR, "Invalid arguments for process mode");
                return 1;
            }
            std::string input_filename = argv[2];
            std::string output_filename = argv[3];
            
            Pipeline pipeline;
            // You can add default filters here or let the user specify them as additional command-line arguments
            pipeline.add_filter(FilterType::GaussianBlur, 2.0f);
            pipeline.add_filter(FilterType::EdgeDetection);
            
            Image input = load_image(input_filename.c_str());
            Image output;
            pipeline.process(input, output);
            save_image(output_filename.c_str(), output);
            free_image(input);
            free_image(output);
            
            std::cout << "Image processing complete." << std::endl;
            Logger::log(LogLevel::INFO, "Image processed: " + input_filename + " -> " + output_filename);
        }
        else if (mode == "benchmark") {
            if (argc != 3) {
                std::cerr << "Benchmark mode usage: " << argv[0] << " benchmark <input_image>" << std::endl;
                Logger::log(LogLevel::ERROR, "Invalid arguments for benchmark mode");
                return 1;
            }
            std::string input_filename = argv[2];
            run_benchmarks(input_filename);
            Logger::log(LogLevel::INFO, "Benchmarks completed for: " + input_filename);
        } else {
            std::cerr << "Invalid mode. Use 'process', 'benchmark', or 'gui'." << std::endl;
            Logger::log(LogLevel::ERROR, "Invalid mode specified: " + mode);
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        Logger::log(LogLevel::ERROR, std::string("Exception caught: ") + e.what());
        return 1;
    }

    Logger::log(LogLevel::INFO, "Application ended successfully");
    Logger::cleanup();
    return 0;
}
