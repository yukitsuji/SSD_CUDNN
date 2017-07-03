#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "softmax.h"
#include "cuda.h"
}

__global__ void addKernel(float *a, float *b)
{
  int i = threadIdx.x;
  b[i] = a[i];

}

void add_sample(float *a, float *b) {
  addKernel<<<1, 16>>>(a, b);
}

void forward_softmax_gpu(softmax_layer sl, float *input_gpu) {
  float alpha = 1.0f;
  float beta = 0.0f;
  CUDNN_CHECK(cudnnSoftmaxForward(cudnn_handler(),
                      CUDNN_SOFTMAX_FAST,
                      CUDNN_SOFTMAX_MODE_INSTANCE,
                      &alpha,
                      sl.inputTensorDesc,
                      input_gpu,
                      &beta,
                      sl.outputTensorDesc,
                      sl.output_gpu));
}
