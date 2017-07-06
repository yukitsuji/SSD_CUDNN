#ifndef POOLING_H_
#define POOLING_H_

#include "alone_net.h"
#include "network.h"

struct pool_layer;
typedef struct pool_layer pool_layer;

struct pool_layer { //TODO: prev_layerをmake_pool_layerの引数に追加する
  LAYER_TYPE type;
  POOL_TYPE p_type;

  float *output;
  float *output_gpu;

  size_t output_size;

  int out_c;
  int out_h;
  int out_w;

  cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc;
  cudnnPoolingDescriptor_t poolDesc;
  cudnnPoolingMode_t p_mode;

  void (*forward_gpu)(struct pool_layer, float *input_gpu);
};

int get_pool_mapsize(int in_size, int kernel_size, int pad, int stride);
pool_layer make_pool_layer_gpu(POOL_TYPE p_type,
                               int batch, int in_c, int in_h, int in_w,
                               int kernel_h, int kernel_w, int pad_h, int pad_w,
                               int stride_h, int stride_w);
void free_pool_layer_gpu(pool_layer pl);
void forward_pool_gpu(pool_layer pl, float *input_gpu);

#endif
