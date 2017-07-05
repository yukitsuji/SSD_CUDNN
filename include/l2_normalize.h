#ifndef NORMALIZE_H_
#define NORMALIZE_H_

#include "alone_net.h"
#include "network.h"

struct normalize_layer;
typedef struct normalize_layer normalize_layer;

struct normalize_layer {
  LAYER_TYPE type;

  float *output;
  float *output_gpu;

  size_t output_size;

  int out_c;
  int out_h;
  int out_w;

  float *scale;
  float *scale_gpu;


  void (*forward_gpu)(struct normalize_layer, float *input_gpu);
};

normalize_layer make_normalize_layer_gpu(int batch, int out_c, int out_h, int out_w);
void free_normalize_layer_gpu(normalize_layer sl);
void forward_normalize_gpu(normalize_layer sl, float *input_gpu);

void add_sample(float *a, float *b);

#endif
