#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"

static int handler = 0;
static cudnnHandle_t cudnnHandle;

cudnnHandle_t cudnn_handler(){
  if (handler == 0) {
    cudnnHandle_t instant_cudnnHandle;
    CUDNN_CHECK(cudnnCreate(&instant_cudnnHandle));
    cudnnHandle = instant_cudnnHandle;
    handler = 1;
  }
  return cudnnHandle;
}

void make_gpu_array(float **x_gpu, float *x, size_t size) {
  CUDA_CHECK(cudaMalloc((void**)x_gpu, size));
  if (x) {
    CUDA_CHECK(cudaMemcpy(*x_gpu, x, size, cudaMemcpyHostToDevice));
  } else {
    float value = 0.0f;
    int block_size = 512;
    set_value_gpu(*x_gpu, size / sizeof(float), value, block_size);
  }
}

void gpu_to_cpu(float *x_gpu, float *x, size_t size) {
  CUDA_CHECK(cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost));
}


dim3 opt_gridsize(size_t size, const int block_size) {
  size_t x = (size - 1) / block_size + 1;
  size_t y = 1;
  dim3 d = {x, y, 1};
  return d;
}
