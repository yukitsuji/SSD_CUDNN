#include <convolution.h>
#include "math_util.h"


int get_conv_mapsize(int in_size, int kernel_size, int pad, int stride) {
  int out_size = (in_size - kernel_size + 2 * pad) / stride + 1;
  return out_size;
}

conv_layer make_conv_layer_gpu(int batch, int out_c, int in_c, int in_h, int in_w,
                               int kernel_h, int kernel_w, int pad_h, int pad_w,
                               int stride_h, int stride_w, int dilation_h, int dilation_w) {
  conv_layer cl;
  cl.type = CONVOLUTION;

  cl.out_c = out_c;
  cl.out_h = get_conv_mapsize(in_c, kernel_h, pad_h, stride_h);
  cl.out_w = get_conv_mapsize(in_c, kernel_w, pad_w, stride_w);

  cl.forward_gpu = forward_conv_gpu;

  cl.output_size = batch * cl.out_c * cl.out_h * cl.out_w * sizeof(float);
  cl.output = calloc(cl.output_size / sizeof(float), sizeof(float));
  make_gpu_array(&cl.output_gpu, 0, cl.output_size);
  cl.weight = calloc(cl.out_c * in_c * kernel_h * kernel_w, sizeof(float));
  int i;
  for(i = 0; i < cl.out_c * in_c * kernel_h * kernel_w; ++i) cl.weight[i] = 0.01 * rand_normal();
  for(i = 0; i < cl.out_c * in_c * kernel_h * kernel_w; ++i) printf("%f\n", cl.weight[i]);
  make_gpu_array(&cl.weight_gpu, cl.weight, cl.out_c * in_c * kernel_h * kernel_w * sizeof(float));

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cl.inputTensorDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cl.outputTensorDesc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&cl.filterDesc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cl.convDesc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(cl.inputTensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            batch, in_c, in_h, in_w));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(cl.outputTensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            batch, cl.out_c, cl.out_h, cl.out_w));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(cl.filterDesc,
                            CUDNN_DATA_FLOAT,
                            CUDNN_TENSOR_NCHW,
                            cl.out_c, in_c, kernel_h, kernel_w));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cl.convDesc,
                            pad_h, pad_w, stride_h, stride_w,
                            dilation_h, dilation_w,
                            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  // CUDA_CHECK(cudaDeviceSynchronize());
  // CUDA_CHECK(cudaGetLastError());
  // cudnnHandle_t instant_cudnnHandle;
  // CUDNN_CHECK(cudnnCreate(&instant_cudnnHandle)); //cudnn_handler(),
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn_handler(),
                            cl.inputTensorDesc,
                            cl.filterDesc,
                            cl.convDesc,
                            cl.outputTensorDesc,
                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                            0,
                            &cl.fw_algo));

  size_t temp_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handler(),
                            cl.inputTensorDesc,
                            cl.filterDesc,
                            cl.convDesc,
                            cl.outputTensorDesc,
                            cl.fw_algo,
                            &temp_size));

  cl.workspace_size = temp_size;
  make_gpu_array(&cl.workspace_gpu, 0, cl.workspace_size);
  CUDA_CHECK(cudaDeviceSynchronize());
  return cl;
}

void free_conv_layer_gpu(conv_layer cl) {
  free(cl.output);
  cudaFree(cl.output_gpu);

  free(cl.weight);
  cudaFree(cl.weight_gpu);

  cudaFree(cl.workspace_gpu);

  CUDNN_CHECK(cudnnDestroyTensorDescriptor(cl.inputTensorDesc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(cl.outputTensorDesc));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(cl.filterDesc));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(cl.convDesc));
}
