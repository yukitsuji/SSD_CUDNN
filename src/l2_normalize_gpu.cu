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

#include <cmath>

// Don't hurt input data.
__global__ void pow_kernel(const float *input, float *output, const unsigned int size, const float alpha) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = id; i < size; i += blockDim.x * gridDimx.x) {
    output[i] = pow(input[i], alpha);
  }
}

__global__ void div_kernel(const float *input, const float *div_input, float *output,
                           const unsigned int size, const int wh) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = id; i < size; i += blockDim.x * gridDimx.x) {
    // input[i] = pow(input[i], alpha);
    output[i] = input[i] / div_input[i / wh];
  }
}

void forward_normalize_gpu(normalize_layer nl, float *input_gpu) {
  float alpha = 1.0f;
  float beta = 0.0f;
  // Sum powed matrix to input channel per Batch
  // Pow 2 for whole batch.
  pow_kernel<<<opt_gridsize(nl.output_size, 512), 512>>>(input_gpu,
               nl.powed_output_gpu, nl.output_size, 2.0f);

  // Caluculate Sum for input/output channel and save. Shape is HW.
  CUBLAS_CHECK(cublasSgemv(cublas_handler(), CUBLAS_OP_N, nl.out_c,
               nl.out_h * nl.out_w, &alpha, nl.powed_output_gpu, nl.out_c,
               nl.ones_channel_gpu, 1, &beta, nl.out_norm_gpu, 1));

  // Pow 1/2 for each batch.B1HW // TODO
  pow_kernel<<<opt_gridsize(nl.output_size, 512), 512>>>(nl.out_norm_gpu,
               nl.out_norm_gpu, nl.out_norm_size, 0.5f);

  // TODO 統合できるかも？
  // divide each batch(CHW) by calculated normalize vector(HW).
  div_kernel<<<opt_gridsize(nl.output_size, 512), 512>>>(input_gpu, nl.out_norm_gpu,

               nl.output_size, nl.out_h * nl.out_w);
  // Scale to whole batch.
  mul_kernel<<<opt_gridsize(nl.output_size, 512), 512>>>();

}
