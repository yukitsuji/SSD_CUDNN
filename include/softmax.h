#ifndef SOFTMAX_H_
#define SOFTMAX_H_

// #include "cuda_util.h"
#include "alone_net.h"
#include "network.h"

struct softmax_layer;
typedef struct softmax_layer softmax_layer;

struct softmax_layer {
  LAYER_TYPE type;

  float *output;
  float *output_gpu;

  size_t output_size;

  int out_c;
  int out_h;
  int out_w;

  cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc;

  void (*forward_gpu)(struct softmax_layer, float *input_gpu);
};

softmax_layer make_softmax_layer_gpu(int batch, int out_c, int out_h, int out_w);
void free_softmax_layer_gpu(softmax_layer sl);
void forward_softmax_gpu(softmax_layer sl, float *input_gpu);

void add_sample(float *a, float *b);

#endif
