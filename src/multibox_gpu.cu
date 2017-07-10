#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "multibox.h"
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

// HWP4 -> HWP4
__global__ void void multibox_decoder_kernel(float *location_gpu, float *priorbox_gpu, float output_gpu, const int size,
                          float variance_xy, float variance_wl){
  const int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
  if (idx >= size) return;
  const float prior_center_x = priorbox_gpu[idx];
  const float prior_center_y = priorbox_gpu[idx + 1];
  const float prior_width = priorbox_gpu[idx + 2];
  const float prior_height = priorbox_gpu[idx + 3];

  const float decode_center_x = variance_xy * location_gpu[idx] * prior_width + prior_center_x;
  const float decode_center_y = variance_xy * location_gpu[idx + 1] * prior_height + prior_center_y;
  const float decode_width = exp(variance_wl * location_gpu[idx + 2]) * prior_width;
  const float decode_height = exp(variance_wl * location_gpu[idx + 3]) * prior_height;

  output_gpu[idx] = decode_center_x - decode_width / 2.; // xmin
  output_gpu[idx + 1] = decode_center_y - decode_height / 2.; // ymin
  output_gpu[idx + 2] = decode_center_x + decode_width / 2.; //xmax
  output_gpu[idx + 3] = decode_center_y + decode_height / 2.; //ymax
}

// HWP -> HWP
void multibox_decoder_gpu(float *location_gpu, float *priorbox_gpu, float output_gpu, const int whp,
                          float variance_xy, float variance_wl){
  const int size = whp * 4;
  multibox_decoder_kernel<<<opt_gridsize(size, 512), 512>>>(location_gpu, priorbox_gpu, output_gpu, size,
                          variance_xy, variance_wl);
}

// HWPC -> HWP        PCHW -> HWP
__global__ void extract_max_kernel(float *output, float *input, int whp, int class_num){
  __shared__ float max_shared[512];
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int x = idx * 21;
  if (idx >= whp) return; // ここでおそらくthreadIdxで過剰な値は消されてるはず……
  max_shared[threadIdx.x] = input[x];
  __syncthreads();
  if (max_shared[threadIdx.x] < input[x + 1]) max_shared[threadIdx.x] = input[x + 1];
  if (max_shared[threadIdx.x] < input[x + 2]) max_shared[threadIdx.x] = input[x + 2];
  if (max_shared[threadIdx.x] < input[x + 3]) max_shared[threadIdx.x] = input[x + 3];
  if (max_shared[threadIdx.x] < input[x + 4]) max_shared[threadIdx.x] = input[x + 4];
  if (max_shared[threadIdx.x] < input[x + 5]) max_shared[threadIdx.x] = input[x + 5];
  if (max_shared[threadIdx.x] < input[x + 6]) max_shared[threadIdx.x] = input[x + 6];
  if (max_shared[threadIdx.x] < input[x + 7]) max_shared[threadIdx.x] = input[x + 7];
  if (max_shared[threadIdx.x] < input[x + 8]) max_shared[threadIdx.x] = input[x + 8];
  if (max_shared[threadIdx.x] < input[x + 9]) max_shared[threadIdx.x] = input[x + 9];
  if (max_shared[threadIdx.x] < input[x + 10]) max_shared[threadIdx.x] = input[x + 10];
  if (max_shared[threadIdx.x] < input[x + 11]) max_shared[threadIdx.x] = input[x + 11];
  if (max_shared[threadIdx.x] < input[x + 12]) max_shared[threadIdx.x] = input[x + 12];
  if (max_shared[threadIdx.x] < input[x + 13]) max_shared[threadIdx.x] = input[x + 13];
  if (max_shared[threadIdx.x] < input[x + 14]) max_shared[threadIdx.x] = input[x + 14];
  if (max_shared[threadIdx.x] < input[x + 15]) max_shared[threadIdx.x] = input[x + 15];
  if (max_shared[threadIdx.x] < input[x + 16]) max_shared[threadIdx.x] = input[x + 16];
  if (max_shared[threadIdx.x] < input[x + 17]) max_shared[threadIdx.x] = input[x + 17];
  if (max_shared[threadIdx.x] < input[x + 18]) max_shared[threadIdx.x] = input[x + 18];
  if (max_shared[threadIdx.x] < input[x + 19]) max_shared[threadIdx.x] = input[x + 19];
  output[idx] = max_shared[threadIdx.x];
}

// PCHW -> HWP
void extract_max_softmax(float *output, float *input, int prior_num, int class_num,
                         int wh){
  const int size = wh * prior_num;
  extract_max_kernel<<<opt_gridsize(size, 512), 512>>>(output, input, size, class_num);
}
