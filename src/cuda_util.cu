#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "cuda_util.h"
#include "cuda.h"
}

__global__ void set_value_kernel(float *x_gpu, const size_t *size, const float *value) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  x_gpu[id] = *value;
}

void set_value_gpu(float **x_gpu, const size_t size, const float value) {
  
  set_value_kernel<<<1, 512>>>(*x_gpu, &size, &value);
}
