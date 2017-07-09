#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "transpose.h"
#include "cuda.h"
}

__global__ void transpose_kernel(float *output, unsigned char *input, int w, int h)
{
  const int sx = blockDim.x * blockIdx.x + threadIdx.x;
  const int sy = blockDim.y * blockIdx.y + threadIdx.y;
  const int out_x = sx * h + sy;
  const int in_x = sy * w * 3 + sx * 3;

  if(w <= sx || h <= sy || h * w <= out_x) return;

  output[out_x] = input[in_x] - 102.9801; // B 102.9801
  output[out_x + w * h] = input[in_x + 1] - 115.9465; // G 115.9465
  output[out_x + 2 * w * h] = input[in_x + 2] - 122.7717; // R 122.7717
}

__global__ void transpose_chw_to_hwc_kernels(float *output, unsigned char *input, int c, int wh)
{
  int size = c * wh;
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx >= size) return;

  output[idx / wh + (idx % wh) * c] = input[idx];
}

void transpose_gpu(float *output, unsigned char *input, int w, int h) {
  int x, y;
  int a;
  dim3 d, d_a;
  a = 16;
  x = (w - 1) / a + 1;
  y = (h - 1) / a + 1;
  dim3 block (x, y, 1);
  dim3 grid  (a, a, 1);
  transpose_kernel<<<block, grid>>>(output, input, w, h);
}

void transpose_chw_to_hwc_gpu(float *output, unsigned char *input, int c, int wh) {
  transpose_kernel<<<opt_gridsize(c*wh, 512), 512>>>(output, input, c, wh);
}
