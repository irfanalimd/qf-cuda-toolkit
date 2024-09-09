#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

void check_cuda_error(cudaError_t err, const char* const func, const char* const file, const int line);

#endif // CUDA_UTILS_H