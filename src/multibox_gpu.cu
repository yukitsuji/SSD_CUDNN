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

// HWC to CHW
__global__ void transpose_chw_to_hwc_kernels(float *output, unsigned char *input, int c, int wh)
{
  int size = c * wh;
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx >= size) return;

  output[idx / wh + (id % wh) * c] = input[idx];
}

void multibox_decoder_gpu(float *location_gpu, float *priorbox_gpu, int all_size,
                          int w, int h, int net_h, int net_w
                          float variance_xy, float variance_wl){
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;

}

void forward_multibox_layer_gpu(multibox_layer nl, float *input_gpu) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // Decoder of location. Need PriorBox and Location map.
  // TransposeしてConcatenateされたPrior BoxとLocation Mapあったほうが一挙に計算しやすい.もしかしたらそっちのほうが速いかも。
  // Outputは、H * W * Prior * 4 * Num_of_Layer
  // Outputは、H * W * Prior * Class * Num_fo_Layer

  // Extract Max Confidence from confidence map.
  // Outputは、H * W * Prior * Num_of_Layer

  // Sum powed matrix to input channel per Batch
  // Pow 2 for whole batch.
  pow_kernel<<<opt_gridsize(nl.output_size, 512), 512>>>(input_gpu,
               nl.powed_output_gpu, nl.output_size / sizeof(float), 2.0f);
}
