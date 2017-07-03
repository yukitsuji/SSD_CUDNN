#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include "alone_net.h"
#include "network.h"

struct conv_layer;
typedef struct conv_layer conv_layer;

struct conv_layer { //TODO: prev_layerをmake_conv_layerの引数に追加する
  LAYER_TYPE type;
  STRUCT_TYPE s_type;
  ACTIVATE_TYPE a_type;

  // Convolution
  float *output;
  float *output_gpu;
  float *weight;
  float *weight_gpu;
  float *workspace_gpu;

  size_t output_size;
  size_t workspace_size;

  int out_c;
  int out_h;
  int out_w;

  cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fw_algo;

  // Bias
  float *bias;
  float *bias_gpu;
  size_t bias_size;
  cudnnTensorDescriptor_t biasTensorDesc;

  // Batch Normalization
  float *bn_input;
  float *bn_input_gpu;
  float *bn_scale;
  float *bn_bias;
  float *bn_result_mean;
  float *bn_result_varience;
  float *bn_scale_gpu;
  float *bn_bias_gpu;
  float *bn_result_mean_gpu;
  float *bn_result_varience_gpu;
  cudnnTensorDescriptor_t bnTensorDesc;

  cudnnActivationDescriptor_t activationDesc;


  void (*forward_gpu)(struct conv_layer, float *input_gpu);
};

int get_conv_mapsize(int in_size, int kernel_size, int pad, int stride);
conv_layer make_conv_layer_gpu(STRUCT_TYPE s_type, ACTIVATE_TYPE a_type, double a_param,
                               int batch, int out_c, int in_c, int in_h, int in_w,
                               int kernel_h, int kernel_w, int pad_h, int pad_w,
                               int stride_h, int stride_w, int dilation_h, int dilation_w);
void free_conv_layer_gpu(conv_layer cl);
void forward_conv_gpu(conv_layer cl, float *input_gpu);

#endif
