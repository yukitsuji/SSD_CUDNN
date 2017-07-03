#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "cuda_util.h"
#include "cuda.h"
}

__global__ void set_value_kernel(float *x_gpu, const int size, const float value) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    x_gpu[id] = value;
  }
}

void set_value_gpu(float *x_gpu, const int size, const float value, const int block_size) {
  set_value_kernel<<<opt_gridsize(size, block_size), block_size>>>(x_gpu, size, value);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
}
