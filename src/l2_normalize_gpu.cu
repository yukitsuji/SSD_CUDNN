#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "l2_normalize.h"
#include "cuda.h"
}

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>


__global__ void addKernel(float *a, float *b)
{
  int i = threadIdx.x;
  b[i] = a[i];

}

void add_sample(float *a, float *b) {
  addKernel<<<1, 16>>>(a, b);
}

void forward_normalize_gpu(normalize_layer nl, float *input_gpu) {
  float alpha = 1.0f;
  float beta = 0.0f;
  // Sum powed matrix to input channel per Batch
  // Pow 2 for whole batch.

  // Caluculate Sum for input/output channel and save. Shape is HW.

  // Pow 1/2 for each batch.

  // TODO 統合できるかも？
  // divide each batch(CHW) by calculated normalize vector(HW).

  // Scale to whole batch.

}
