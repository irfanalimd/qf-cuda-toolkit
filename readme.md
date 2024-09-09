# QuantumFilter: CUDA-Enhanced Image Processing Toolkit

## Overview

This CUDA-based image processing library provides high-performance image manipulation tools leveraging GPU acceleration. It's designed for efficient processing of large images or batch processing tasks.

## Features

- GPU-accelerated image processing using CUDA
- Multiple filter implementations:
  - Gaussian Blur
  - Sobel Edge Detection
  - Brightness and Contrast Adjustment
- Flexible pipeline for chaining multiple filters
- Benchmarking tools to compare GPU vs CPU performance
- Command-line interface for easy integration into workflows

## Requirements

- CUDA-capable GPU
- CUDA Toolkit (version 11.0 or higher)
- C++14 compatible compiler
- CMake (version 3.10 or higher)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/irfanalimd/qf-cuda-toolkit.git
   cd qf-cuda-toolkit
   ```

2. Create a build directory and navigate to it:
   ```
   mkdir build && cd build
   ```

3. Configure the project with CMake:
   ```
   cmake ..
   ```

4. Build the project:
   ```
   cmake --build .
   ```

## Usage

The application supports two modes: process and benchmark.

### Process Mode

To process an image:

```
./image_processor process <input_image_path> <output_image_path>
```

Example:
```
./image_processor process input.jpg output.jpg
```

### Benchmark Mode

To run benchmarks comparing GPU and CPU performance:

```
./image_processor benchmark <input_image_path>
```

Example:
```
./image_processor benchmark test_image.png
```

## Customizing the Pipeline

To customize the image processing pipeline, modify the `process_image` function in `src/pipeline.cu`. You can add or remove filters and adjust their parameters.

Example:
```cpp
void process_image(const char* input_filename, const char* output_filename) {
    Image input = load_image(input_filename);
    Image output;

    Pipeline pipeline;
    pipeline.add_filter(FilterType::GaussianBlur, 2.0f);
    pipeline.add_filter(FilterType::EdgeDetection);
    pipeline.add_filter(FilterType::BrightnessContrast, 0.1f, 1.2f);

    pipeline.process(input, output);

    save_image(output_filename, output);
    free_image(input);
    free_image(output);
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.