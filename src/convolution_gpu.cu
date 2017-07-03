#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolution.h"
#include "cuda.h"
}

void forward_conv_gpu(conv_layer cl, float *input_gpu) {
  float alpha = 1.0f;
  CUDNN_CHECK(cudnnConvolutionForward(cudnn_handler(),
                      &alpha,
                      cl.inputTensorDesc,
                      input_gpu,
                      cl.filterDesc,
                      cl.weight_gpu,
                      cl.convDesc,
                      cl.fw_algo,
                      cl.workspace_gpu,
                      cl.workspace_size,
                      &alpha,
                      cl.outputTensorDesc,
                      cl.output_gpu));
}
