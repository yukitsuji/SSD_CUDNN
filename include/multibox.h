#ifndef NORMALIZE_H_
#define NORMALIZE_H_

#include "alone_net.h"
#include "network.h"

struct multibox_layer;
typedef struct multibox_layer multibox_layer;

struct multibox_layer {
  LAYER_TYPE type;

  float *output;
  float *output_gpu;

  size_t output_size;

  int out_c;
  int out_h;
  int out_w;

  float *scale;
  float *scale_gpu;
  size_t scale_size;

  float *out_norm;
  float *out_norm_gpu;
  size_t out_norm_size;

  float *powed_output;
  float *powed_output_gpu;

  float *ones_channel;
  float *ones_channel_gpu;
  size_t ones_channel_size;

  void (*forward_gpu)(struct multibox_layer, float *input_gpu);
};

multibox_layer make_multibox_layer_gpu(int batch, int out_c, int out_h, int out_w, int scale);
void free_multibox_layer_gpu(multibox_layer ml);
void forward_normalize_gpu(multibox_layer ml, float *input_gpu);
void multibox_decoder_gpu(float *location_gpu, float *priorbox_gpu, float output_gpu, const int whp,
                          float variance_xy, float variance_wl);
void extract_max_softmax(float *output, float *input, int prior_num, int class_num,
                         int wh);

#endif
