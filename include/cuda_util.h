#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"

static const char *cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "<unknown>";
    }
}

// static void cuda_status_check(cudaError_t status, const char *file, int line);
// static void cudnn_status_check(cudnnStatus_t status, const char *file, int line);
static void cuda_status_check(cudaError_t status, const char *file, int line) {
  cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    fprintf(stderr, "cuda error [%d] : %s. in File name %s at Line %d\n", status,
            cudaGetErrorString(status), file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  } else {
    fprintf(stderr, "Success [%d]. file is %s, Line is %d\n", status, file, line);
  }
}

static void cudnn_status_check(cudnnStatus_t status, const char *file, int line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    fprintf(stderr, "cudnn error [%d] : %s. in File name %s at Line %d\n", status,
            cudnnGetErrorString(status), file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  } else {
    fprintf(stderr, "Success [%d]. file is %s, Line is %d\n", status, file, line);
  }
}

static void cublas_status_check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublas error [%d] : %s. in File name %s at Line %d\n", status,
            cublasGetErrorString(status), file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  } else {
    fprintf(stderr, "cublas Success [%d]. file is %s, Line is %d\n", status, file, line);
  }
}

#define CUDA_CHECK(status) (cuda_status_check(status, __FILE__, __LINE__));
#define CUDNN_CHECK(status) (cudnn_status_check(status, __FILE__, __LINE__));
#define CUBLAS_CHECK(status) (cublas_status_check(status, __FILE__, __LINE__));
// #define KERNEL_CHECK()

cudnnHandle_t cudnn_handler();
dim3 opt_gridsize(size_t size, const int block_size);
void set_value_gpu(float *x_gpu, const int size, const float value, const int block_size);
void make_gpu_array(float **x_gpu, float *x, size_t size);
void gpu_to_cpu(float *x_gpu, float *x, size_t size);

#endif
