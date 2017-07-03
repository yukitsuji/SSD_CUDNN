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

  if (cl.s_type == BIAS) {
    CUDNN_CHECK(cudnnAddTensor(cudnn_handler(),
                      &alpha,
                      cl.biasTensorDesc,
                      cl.bias_gpu,
                      &alpha,
                      cl.outputTensorDesc,
                      cl.output_gpu));
  }

  if (cl.s_type == BATCH_NORM) {
    float one = 1;
    float zero = 0;
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(cudnn_handler(),
                      CUDNN_BATCHNORM_SPATIAL,
                      &one,
                      &zero,
                      cl.outputTensorDesc,
                      cl.output_gpu,
                      cl.outputTensorDesc,
                      cl.output_gpu,
                      cl.bnTensorDesc,
                      cl.bn_scale_gpu,
                      cl.bn_bias_gpu,
                      cl.bn_result_mean_gpu,
                      cl.bn_result_varience_gpu,
                      .00001));
  }

  if (cl.a_type != NONE_S) {
    float one = 1;
    float zero = 0;
    CUDNN_CHECK(cudnnActivationForward(cudnn_handler(),
                      cl.activationDesc,
                      &one,
                      cl.outputTensorDesc,
                      cl.output_gpu,
                      &zero,
                      cl.outputTensorDesc,
                      cl.output_gpu))
  }
}
