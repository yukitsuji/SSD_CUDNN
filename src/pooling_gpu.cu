#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "pooling.h"
#include "cuda.h"
}

void forward_pool_layer_gpu(pool_layer pl, float *input_gpu) {
  float alpha = 1.0f;
  float beta = 0.0f;
  CUDNN_CHECK(cudnnPoolingForward(cudnn_handler(),
                          pl.poolDesc,
                          &alpha,
                          pl.inputTensorDesc,
                          input_gpu,
                          &beta,
                          pl.outputTensorDesc,
                          pl.output_gpu));
}
