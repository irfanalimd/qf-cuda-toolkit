#include "pipeline.h"
#include "filters/optical_flow.h"
#include "benchmark.h"
#include "logger.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// Convert OpenCV Mat to Image struct
Image mat_to_image(const cv::Mat& mat) {
    Image img;
    img.width = mat.cols;
    img.height = mat.rows;
    img.channels = mat.channels();
    img.data = new unsigned char[img.width * img.height * img.channels];
    
    if (mat.isContinuous()) {
        memcpy(img.data, mat.data, img.width * img.height * img.channels);
    } else {
        for (int y = 0; y < img.height; ++y) {
            memcpy(img.data + y * img.width * img.channels,
                   mat.ptr(y), img.width * img.channels);
        }
    }
    
    return img;
}

// Convert Image struct to OpenCV Mat
cv::Mat image_to_mat(const Image& img) {
    cv::Mat mat(img.height, img.width, 
                img.channels == 3 ? CV_8UC3 : CV_8UC1,
                img.data);
    return mat.clone();
}

void run_optical_flow_demo(int camera_id = 0) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return;
    }
    
    // Set camera properties for better performance
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Optical Flow", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Flow Visualization", cv::WINDOW_AUTOSIZE);
    
    Image prev_frame, curr_frame;
    bool first_frame = true;
    
    OpticalFlowParams flow_params;
    flow_params.window_size = 15;
    flow_params.min_eigenvalue = 0.001f;
    
    // FPS calculation
    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double fps = 0.0;
    
    std::cout << "Press 'q' to quit, 's' to save current flow visualization" << std::endl;
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) break;
        
        // Convert to RGB (OpenCV uses BGR)
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        
        curr_frame = mat_to_image(frame);
        
        if (!first_frame) {
            // Allocate flow field
            FlowVector* flow = new FlowVector[curr_frame.width * curr_frame.height];
            
            // Compute optical flow
            auto start = std::chrono::high_resolution_clock::now();
            compute_optical_flow_gpu(prev_frame, curr_frame, flow, flow_params);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double ms = duration.count() / 1000.0;
            
            // Create flow visualization
            Image flow_viz;
            flow_to_color(flow, curr_frame.width, curr_frame.height, flow_viz);
            
            // Display results
            cv::Mat orig_bgr;
            cv::cvtColor(frame, orig_bgr, cv::COLOR_RGB2BGR);
            cv::imshow("Original", orig_bgr);
            
            cv::Mat flow_mat = image_to_mat(flow_viz);
            cv::cvtColor(flow_mat, flow_mat, cv::COLOR_RGB2BGR);
            
            // Add text overlay with FPS
            cv::putText(flow_mat, cv::format("GPU Time: %.2f ms", ms), 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cv::Scalar(255, 255, 255), 2);
            cv::putText(flow_mat, cv::format("FPS: %.1f", fps), 
                       cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cv::Scalar(255, 255, 255), 2);
            
            cv::imshow("Optical Flow", flow_mat);
            
            // Arrow visualization
            Image arrow_viz;
            visualize_flow_field(flow, curr_frame.width, curr_frame.height, arrow_viz, 2.0f);
            cv::Mat arrow_mat = image_to_mat(arrow_viz);
            cv::cvtColor(arrow_mat, arrow_mat, cv::COLOR_RGB2BGR);
            cv::imshow("Flow Visualization", arrow_mat);
            
            delete[] flow;
            free_image(flow_viz);
            free_image(arrow_viz);
            
            // Update FPS
            frame_count++;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - last_time);
            if (time_span.count() > 1.0) {
                fps = frame_count / time_span.count();
                frame_count = 0;
                last_time = current_time;
            }
        }
        
        // Clean up previous frame
        if (!first_frame) {
            free_image(prev_frame);
        }
        
        prev_frame = curr_frame;
        first_frame = false;
        
        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;  // 'q' or ESC
        if (key == 's') {
            cv::Mat flow_mat = cv::getWindowImageRect("Optical Flow").size() == cv::Size(0,0) ? 
                              cv::Mat() : cv::getWindowImageRect("Optical Flow");
            if (!flow_mat.empty()) {
                cv::imwrite("optical_flow_output.png", flow_mat);
                std::cout << "Saved flow visualization to optical_flow_output.png" << std::endl;
            }
        }
    }
    
    // Cleanup
    if (!first_frame) {
        free_image(prev_frame);
    }
    
    cap.release();
    cv::destroyAllWindows();
}

void benchmark_optical_flow(const std::string& image1_path, const std::string& image2_path) {
    Logger::log(LogLevel::INFO, "Starting optical flow benchmark");
    
    Image img1 = load_image(image1_path.c_str());
    Image img2 = load_image(image2_path.c_str());
    
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "Error: Images must have the same dimensions" << std::endl;
        free_image(img1);
        free_image(img2);
        return;
    }
    
    FlowVector* flow_gpu = new FlowVector[img1.width * img1.height];
    FlowVector* flow_cpu = new FlowVector[img1.width * img1.height];
    
    OpticalFlowParams params;
    params.window_size = 15;
    
    Timer timer;
    
    // Warm-up GPU
    compute_optical_flow_gpu(img1, img2, flow_gpu, params);
    
    // Benchmark GPU
    timer.start();
    for (int i = 0; i < 10; ++i) {
        compute_optical_flow_gpu(img1, img2, flow_gpu, params);
    }
    timer.stop();
    double gpu_time = timer.elapsed() / 10.0;
    
    // Benchmark CPU
    timer.start();
    compute_optical_flow_cpu(img1, img2, flow_cpu, params);
    timer.stop();
    double cpu_time = timer.elapsed();
    
    double speedup = cpu_time / gpu_time;
    
    std::cout << "\n=== Optical Flow Benchmark Results ===" << std::endl;
    std::cout << "Image size: " << img1.width << "x" << img1.height << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    Logger::log(LogLevel::INFO, "Optical Flow - GPU: " + std::to_string(gpu_time) + 
                "ms, CPU: " + std::to_string(cpu_time) + "ms, Speedup: " + std::to_string(speedup) + "x");
    
    // Save visualization
    Image flow_viz;
    flow_to_color(flow_gpu, img1.width, img1.height, flow_viz);
    save_image("optical_flow_result.png", flow_viz);
    
    delete[] flow_gpu;
    delete[] flow_cpu;
    free_image(img1);
    free_image(img2);
    free_image(flow_viz);
}

int main(int argc, char** argv) {
    Logger::init("optical_flow.log");
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> [args]" << std::endl;
        std::cerr << "Modes:" << std::endl;
        std::cerr << "  demo - Run real-time webcam demo" << std::endl;
        std::cerr << "  benchmark <img1> <img2> - Benchmark optical flow between two images" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    
    if (mode == "demo") {
        run_optical_flow_demo();
    } else if (mode == "benchmark") {
        if (argc != 4) {
            std::cerr << "Benchmark mode requires two image paths" << std::endl;
            return 1;
        }
        benchmark_optical_flow(argv[2], argv[3]);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
    
    Logger::cleanup();
    return 0;
}