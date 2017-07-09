#ifndef NETWORK_H_
#define NETWORK_H_

#include "cuda_util.h"

typedef enum {
  CONVOLUTION,
  SOFTMAX,
  POOLING,
  NORMALIZE,
  MULTIBOX,
} LAYER_TYPE;

typedef enum {
  NONE_S,
  BIAS,
  BATCH_NORM,
} STRUCT_TYPE;

typedef enum {
  NONE_A,
  SIGMOID,
  RELU,
  TANH,
  CLIPPED_RELU,
  ELU,
} ACTIVATE_TYPE;

typedef enum {
  MAX_POOL,
  AVG_IN,
  AVG_EX,
  MAX_DETERM,
} POOL_TYPE;

struct base_network {

};
#endif
