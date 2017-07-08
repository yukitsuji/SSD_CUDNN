#include <pooling.h>
#include "math_util.h"


int get_pool_mapsize(int in_size, int kernel_size, int pad, int stride) {
  int out_size = (in_size - kernel_size + 2 * pad) / stride + 1;
  return out_size;
}

pool_layer make_pool_layer_gpu(POOL_TYPE p_type,
                               int batch, int in_c, int in_h, int in_w,
                               int kernel_h, int kernel_w, int pad_h, int pad_w,
                               int stride_h, int stride_w) {
  pool_layer pl;
  pl.type = POOLING;
  pl.p_type = p_type;

  pl.out_c = in_c;
  pl.out_h = get_pool_mapsize(in_c, kernel_h, pad_h, stride_h);
  pl.out_w = get_pool_mapsize(in_c, kernel_w, pad_w, stride_w);

  pl.forward_gpu = forward_pool_gpu;

  pl.output_size = batch * pl.out_c * pl.out_h * pl.out_w * sizeof(float);
  pl.output = calloc(pl.output_size / sizeof(float), sizeof(float));
  make_gpu_array(&pl.output_gpu, 0, pl.output_size);

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&pl.inputTensorDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&pl.outputTensorDesc));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pl.poolDesc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(pl.inputTensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            batch, in_c, in_h, in_w));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(pl.outputTensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            batch, pl.out_c, pl.out_h, pl.out_w));

  if (pl.p_type == MAX_POOL) {
    pl.p_mode = CUDNN_POOLING_MAX;
  } else if (pl.p_type == AVG_IN) {
    pl.p_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else if (pl.p_type == AVG_EX) {
    pl.p_mode =CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  } else if (pl.p_type == MAX_DETERM) {
    pl.p_mode = CUDNN_POOLING_MAX_DETERMINISTIC;
  }

  CUDNN_CHECK(cudnnSetPooling2dDescriptor(pl.poolDesc,
                            pl.p_mode,
                            CUDNN_PROPAGATE_NAN,
                            kernel_h,
                            kernel_w,
                            pad_h,
                            pad_w,
                            stride_h,
                            stride_w));
  CUDA_CHECK(cudaDeviceSynchronize());

  return pl;
}

void free_pool_layer_gpu(pool_layer pl) {
  free(pl.output);
  cudaFree(pl.output_gpu);

  CUDNN_CHECK(cudnnDestroyTensorDescriptor(pl.inputTensorDesc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(pl.outputTensorDesc));
  CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pl.poolDesc));
}
