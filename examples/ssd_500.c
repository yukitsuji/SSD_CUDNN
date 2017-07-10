#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include "convolution.h"
#include "softmax.h"
#include "pooling.h"
#include "transpose.h"
#include "l2_normalize.h"
#include "math_util.h"
#include "multibox.h"


int main(void){
  float* input = calloc(16, sizeof(float));
  float* output = calloc(16, sizeof(float));

  int i;
  for (i=0; i < 16; ++i) {
    input[i] = (float)i;
  }

  CUDA_CHECK(cudaSetDevice(0));
  float *input_gpu;

  size_t size = sizeof(float) * 16;
  make_gpu_array(&input_gpu, input, size);

  int batch = 2;
  int input_h = 512;
  int input_w = 512;
  // conv_1_1 relu
  conv_layer conv1_1 = make_conv_layer_gpu(1, 2, 0, batch, 64, 3, 512, 512,
                                         3, 3, 1, 1,
                                         1, 1, 1, 1);

  conv_layer conv1_2 = make_conv_layer_gpu(1, 2, 0, batch, 64,
                                        conv1_1.out_c, conv1_1.out_h, conv1_1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  pool_layer pool1 = make_pool_layer_gpu(0,
                                         batch, 2, 2, 2,
                                         2, 2, 0, 0,
                                         1, 1);

  conv_layer conv2_1 = make_conv_layer_gpu(1, 2, 0, batch, 128,
                                        pool1.out_c, pool1.out_h, pool1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  conv_layer conv2_2 = make_conv_layer_gpu(1, 2, 0, batch, 128,
                                        conv2_1.out_c, conv2_1.out_h, conv2_1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  pool_layer pool2 = make_pool_layer_gpu(0,
                                         batch, 2, 2, 2,
                                         2, 2, 0, 0,
                                         1, 1);

  conv_layer conv3_1 = make_conv_layer_gpu(1, 2, 0, batch, 256,
                                        pool2.out_c, pool2.out_h, pool2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  conv_layer conv3_2 = make_conv_layer_gpu(1, 2, 0, batch, 256,
                                        conv3_1.out_c, conv3_1.out_h, conv3_1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  conv_layer conv3_3 = make_conv_layer_gpu(1, 2, 0, batch, 256,
                                        conv3_2.out_c, conv3_2.out_h, conv3_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  pool_layer pool3 = make_pool_layer_gpu(0,
                                         batch, 2, 2, 2,
                                         2, 2, 0, 0,
                                         1, 1);

  conv_layer conv4_1 = make_conv_layer_gpu(1, 2, 0, batch, 512,
                                        pool3.out_c, pool3.out_h, pool3.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  conv_layer conv4_2 = make_conv_layer_gpu(1, 2, 0, batch, 512,
                                        conv4_1.out_c, conv4_1.out_h, conv4_1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  conv_layer conv4_3 = make_conv_layer_gpu(1, 2, 0, batch, 512,
                                        conv4_2.out_c, conv4_2.out_h, conv4_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  pool_layer pool4 = make_pool_layer_gpu(0,
                                         batch, 2, 2, 2,
                                         2, 2, 0, 0,
                                         1, 1);

  // CONV5
  conv_layer conv5_1 = make_conv_layer_gpu(1, 2, 0, batch, 512,
                                        pool4.out_c, pool4.out_h, pool4.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  conv_layer conv5_2 = make_conv_layer_gpu(1, 2, 0, batch, 512,
                                        conv5_1.out_c, conv5_1.out_h, conv5_1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  conv_layer conv5_3 = make_conv_layer_gpu(1, 2, 0, batch, 512,
                                        conv5_2.out_c, conv5_2.out_h, conv5_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  pool_layer pool5 = make_pool_layer_gpu(0,
                                         batch, 2, 2, 2,
                                         2, 2, 0, 0,
                                         1, 1);

  // FC6
  conv_layer fc6 = make_conv_layer_gpu(1, 2, 0, batch, 1024,
                                        pool5.out_c, pool5.out_h, pool5.out_w,
                                        3, 3, 6, 6,
                                        1, 1, 6, 6);

  // FC7
  conv_layer fc7 = make_conv_layer_gpu(1, 2, 0, batch, 1024,
                                        fc6.out_c, fc6.out_h, fc6.out_w,
                                        1, 1, 0, 0,
                                        1, 1, 1, 1);

  // CONV6
  conv_layer conv6_1 = make_conv_layer_gpu(1, 2, 0, batch, 256,
                                        fc7.out_c,  fc7.out_h,  fc7.out_w,
                                        1, 1, 0, 0,
                                        1, 1, 1, 1);
  conv_layer conv6_2 = make_conv_layer_gpu(1, 2, 0, batch, 512,
                                        conv6_1.out_c, conv6_1.out_h, conv6_1.out_w,
                                        3, 3, 1, 1,
                                        2, 2, 1, 1);

  // CONV7
  conv_layer conv7_1 = make_conv_layer_gpu(1, 2, 0, batch, 128,
                                        conv6_2.out_c,  conv6_2.out_h,  conv6_2.out_w,
                                        1, 1, 0, 0,
                                        1, 1, 1, 1);
  conv_layer conv7_2 = make_conv_layer_gpu(1, 2, 0, batch, 256,
                                        conv7_1.out_c, conv7_1.out_h, conv7_1.out_w,
                                        3, 3, 1, 1,
                                        2, 2, 1, 1);

  // CONV8
  conv_layer conv8_1 = make_conv_layer_gpu(1, 2, 0, batch, 128,
                                        conv7_2.out_c,  conv7_2.out_h,  conv7_2.out_w,
                                        1, 1, 0, 0,
                                        1, 1, 1, 1);
  conv_layer conv8_2 = make_conv_layer_gpu(1, 2, 0, batch, 256,
                                        conv8_1.out_c, conv8_1.out_h, conv8_1.out_w,
                                        3, 3, 1, 1,
                                        2, 2, 1, 1);


  // CONV9
  conv_layer conv9_1 = make_conv_layer_gpu(1, 2, 0, batch, 128,
                                        conv8_2.out_c,  conv8_2.out_h,  conv8_2.out_w,
                                        1, 1, 0, 0,
                                        1, 1, 1, 1);
  conv_layer conv9_2 = make_conv_layer_gpu(1, 2, 0, batch, 256,
                                        conv9_1.out_c, conv9_1.out_h, conv9_1.out_w,
                                        3, 3, 1, 1,
                                        2, 2, 1, 1);

  // CONV9
  conv_layer conv10_1 = make_conv_layer_gpu(1, 2, 0, batch, 128,
                                        conv8_2.out_c,  conv8_2.out_h,  conv8_2.out_w,
                                        1, 1, 0, 0,
                                        1, 1, 1, 1);
  conv_layer conv10_2 = make_conv_layer_gpu(1, 2, 0, batch, 256,
                                        conv10_1.out_c, conv10_1.out_h, conv10_1.out_w,
                                        4, 4, 1, 1,
                                        1, 1, 1, 1);

  normalize_layer norm4_3 = make_norm_layer_gpu(batch, conv4_3.out_c, conv4_3.out_h, conv4_3.out_w);

  ///////////////////////////////////////////
  ////////////      OUTPUT    ///////////////
  ///////////////////////////////////////////
  // CONV4_3_norm_loc
  conv_layer conv4_3_norm_loc = make_conv_layer_gpu(0, 0, 0, batch, 16,
                                        norm4_3.out_c,  norm4_3.out_h,  norm4_3.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  // CONV4_3_norm_conf
  conv_layer conv4_3_norm_conf = make_conv_layer_gpu(0, 0, 0, batch, 84,
                                        norm4_3.out_c,  norm4_3.out_h,  norm4_3.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);

  // FC7_loc
  conv_layer fc7_loc = make_conv_layer_gpu(0, 0, 0, batch, 24,
                                        fc7.out_c,  fc7.out_h,  fc7.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  // FC7_conf
  conv_layer fc7_conf = make_conv_layer_gpu(0, 0, 0, batch, 126,
                                        fc7.out_c,  fc7.out_h,  fc7.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);

  // CONV6_2_loc
  conv_layer conv6_2_loc = make_conv_layer_gpu(0, 0, 0, batch, 24,
                                        conv6_2.out_c,  conv6_2.out_h,  conv6_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  // CONV6_2_conf
  conv_layer conv6_2_conf = make_conv_layer_gpu(0, 0, 0, batch, 126,
                                        conv6_2.out_c,  conv6_2.out_h,  conv6_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);

  // CONV7_2_loc
  conv_layer conv7_2_loc = make_conv_layer_gpu(0, 0, 0, batch, 24,
                                        conv7_2.out_c,  conv7_2.out_h,  conv7_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  // CONV7_2_conf
  conv_layer conv7_2_conf = make_conv_layer_gpu(0, 0, 0, batch, 126,
                                        conv7_2.out_c,  conv7_2.out_h,  conv7_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);

  // CONV8_2_loc
  conv_layer conv8_2_loc = make_conv_layer_gpu(0, 0, 0, batch, 24,
                                        conv8_2.out_c,  conv8_2.out_h,  conv8_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  // CONV8_2_conf
  conv_layer conv8_2_conf = make_conv_layer_gpu(0, 0, 0, batch, 126,
                                        conv8_2.out_c,  conv8_2.out_h,  conv8_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);

  // CONV9_2_loc
  conv_layer conv9_2_loc = make_conv_layer_gpu(0, 0, 0, batch, 16,
                                        conv9_2.out_c,  conv9_2.out_h,  conv9_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  // CONV9_2_conf
  conv_layer conv9_2_conf = make_conv_layer_gpu(0, 0, 0, batch, 84,
                                        conv9_2.out_c,  conv9_2.out_h,  conv9_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);

  // CONV10_2_loc
  conv_layer conv10_2_loc = make_conv_layer_gpu(0, 0, 0, batch, 16,
                                        conv10_2.out_c,  conv10_2.out_h,  conv10_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  // CONV10_2_conf
  conv_layer conv10_2_conf = make_conv_layer_gpu(0, 0, 0, batch, 84,
                                        conv10_2.out_c,  conv10_2.out_h,  conv10_2.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);


  softmax_layer softmax4_3_conf = make_softmax_gpu(batch, conv4_3_norm_conf.out_c,
                                        conv4_3_norm_conf.out_h, conv4_3_norm_conf.out_w);

  softmax_layer softmax_fc7_conf = make_softmax_gpu(batch, fc7_conf.out_c,
                                        fc7_conf.out_h, fc7_conf.out_w);

  softmax_layer softmax6_2_conf = make_softmax_gpu(batch, conv6_2_norm_conf.out_c,
                                        conv6_2_norm_conf.out_h, conv6_2_norm_conf.out_w);

  softmax_layer softmax7_2_conf = make_softmax_gpu(batch, conv7_2_norm_conf.out_c,
                                        conv7_2_norm_conf.out_h, conv7_2_norm_conf.out_w);

  softmax_layer softmax8_2_conf = make_softmax_gpu(batch, conv8_2_norm_conf.out_c,
                                        conv8_2_norm_conf.out_h, conv8_2_norm_conf.out_w);

  softmax_layer softmax9_2_conf = make_softmax_gpu(batch, conv9_2_norm_conf.out_c,
                                        conv9_2_norm_conf.out_h, conv9_2_norm_conf.out_w);

  softmax_layer softmax10_2_conf = make_softmax_gpu(batch, conv10_2_norm_conf.out_c,
                                        conv10_2_norm_conf.out_h, conv10_2_norm_conf.out_w);

  // ここまで、NCHW
  int concat_size;
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////         Prior Box     /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  float *concat_prior_data;
  float *concat_prior_data_gpu;      // HWP4
  int all_concat_prior_size;

  all_concat_prior_size = conv4_3_norm_loc.output_size / sizeof(float)
                        + softmax_fc7_loc.output_size / sizeof(float)
                        + softmax6_2_loc.output_size / sizeof(float)
                        + softmax7_2_loc.output_size / sizeof(float)
                        + softmax8_2_loc.output_size / sizeof(float)
                        + softmax9_2_loc.output_size / sizeof(float)
                        + softmax10_2_loc.output_size / sizeof(float);

  concat_size = 0;
  make_priorbox(concat_prior_data + concat_size, conv4_3_norm_loc.out_h, conv4_3_norm_loc.out_w,
                input_h, input_w, 20.48, 51.2, 0); // Create PriorBox
  concat_size += conv4_3_norm_loc.output_size / sizeof(float);

  make_priorbox(concat_prior_data + concat_size, softmax_fc7_loc.out_h, softmax_fc7_loc.out_w,
                input_h, input_w, 51.2, 133.12, 1); // Create PriorBox
  concat_size += softmax_fc7_loc.output_size / sizeof(float);

  make_priorbox(concat_prior_data + concat_size, softmax6_2_loc.out_h, softmax6_2_loc.out_w,
                input_h, input_w, 133.12, 215.04, 1); // Create PriorBox
  concat_size += softmax6_2_loc.output_size / sizeof(float);

  make_priorbox(concat_prior_data + concat_size, softmax7_2_loc.out_h, softmax7_2_loc.out_w,
                input_h, input_w, 215.04, 296.96, 1); // Create PriorBox
  concat_size += softmax7_2_loc.output_size / sizeof(float);

  make_priorbox(concat_prior_data + concat_size, softmax8_2_loc.out_h, softmax8_2_loc.out_w,
                input_h, input_w, 296.96, 378.88, 1); // Create PriorBox
  concat_size += softmax8_2_loc.output_size / sizeof(float);

  make_priorbox(concat_prior_data + concat_size, softmax9_2_loc.out_h, softmax9_2_loc.out_w,
                input_h, input_w, 378.88, 460.8, 0); // Create PriorBox
  concat_size += softmax9_2_loc.output_size / sizeof(float);

  make_priorbox(concat_prior_data + concat_size, softmax10_2_loc.out_h, softmax10_2_loc.out_w,
                input_h, input_w, 460.8, 542.72, 0); // Create PriorBox
  concat_size += softmax10_2_loc.output_size / sizeof(float);
  make_gpu_array(&concat_prior_data_gpu, concat_prior_data, concat_size*sizeof(float));

  ///////////////////////////////////////////////////
  ///////////  Concatenate location data  ///////////
  ///////////////////////////////////////////////////
  float *concat_location_data_gpu;   // HWP4
  int all_concat_location_size;
  all_concat_location_size = conv4_3_norm_loc.output_size / sizeof(float)
                           + softmax_fc7_loc.output_size / sizeof(float)
                           + softmax6_2_loc.output_size / sizeof(float)
                           + softmax7_2_loc.output_size / sizeof(float)
                           + softmax8_2_loc.output_size / sizeof(float)
                           + softmax9_2_loc.output_size / sizeof(float)
                           + softmax10_2_loc.output_size / sizeof(float);
  make_gpu_array(&concat_location_data_gpu, 0, all_concat_location_size*sizeof(float));

  concat_size = 0;
  transpose_chw_to_hwc_gpu(concat_location_data_gpu + concat_size, conv4_3_norm_loc.output_gpu,
      conv4_3_norm_loc.out_c, conv4_3_norm_loc.out_h * conv4_3_norm_loc.out_w);
  concat_size += conv4_3_norm_loc.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_location_data_gpu + concat_size, softmax_fc7_loc.output_gpu,
      softmax_fc7_loc.out_c, softmax_fc7_loc.out_h * softmax_fc7_loc.out_w)
  concat_size += softmax_fc7_loc.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_location_data_gpu + concat_size, softmax6_2_loc.output_gpu,
      softmax6_2_loc.out_c, softmax6_2_loc.out_h * softmax6_2_loc.out_w)
  concat_size += softmax6_2_loc.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_location_data_gpu + concat_size, softmax7_2_loc.output_gpu,
      softmax7_2_loc.out_c, softmax7_2_loc.out_h * softmax7_2_loc.out_w)
  concat_size += softmax7_2_loc.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_location_data_gpu + concat_size, softmax8_2_loc.output_gpu,
      softmax8_2_loc.out_c, softmax8_2_loc.out_h * softmax8_2_loc.out_w)
  concat_size += softmax8_2_loc.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_location_data_gpu + concat_size, softmax9_2_loc.output_gpu,
      softmax9_2_loc.out_c, softmax9_2_loc.out_h * softmax9_2_loc.out_w)
  concat_size += softmax9_2_loc.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_location_data_gpu + concat_size, softmax10_2_loc.output_gpu,
      softmax10_2_loc.out_c, softmax10_2_loc.out_h * softmax10_2_loc.out_w)
  concat_size += softmax10_2_loc.output_size / sizeof(float);


  ///////////////////////////////////////////////////
  ///////////  Concatenated Confidence data  ////////
  ///////////////////////////////////////////////////
  float *concat_confidence_data_gpu; // HWPC
  float *concat_final_confidence_data_gpu; // HWP
  int class_num = 21;
  all_concat_confidence_size = conv4_3_norm_conf.output_size / sizeof(float)
                           + softmax_fc7_conf.output_size / sizeof(float)
                           + softmax6_2_conf.output_size / sizeof(float)
                           + softmax7_2_conf.output_size / sizeof(float)
                           + softmax8_2_conf.output_size / sizeof(float)
                           + softmax9_2_conf.output_size / sizeof(float)
                           + softmax10_2_conf.output_size / sizeof(float);
  // HWPC
  make_gpu_array(&concat_confidence_data_gpu, 0, all_concat_confidence_size*sizeof(float));
  // HWP
  make_gpu_array(&concat_final_confidence_data_gpu, 0, all_concat_confidence_size*sizeof(float) / class_num);

  concat_size = 0;
  transpose_chw_to_hwc_gpu(concat_confidence_data_gpu + concat_size, conv4_3_norm_conf.output_gpu,
      conv4_3_norm_conf.out_c, conv4_3_norm_conf.out_h * conv4_3_norm_conf.out_w);
  concat_size += conv4_3_norm_conf.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_confidence_data_gpu + concat_size, softmax_fc7_conf.output_gpu,
      softmax_fc7_conf.out_c, softmax_fc7_conf.out_h * softmax_fc7_conf.out_w)
  concat_size += softmax_fc7_conf.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_confidence_data_gpu + concat_size, softmax6_2_conf.output_gpu,
      softmax6_2_conf.out_c, softmax6_2_conf.out_h * softmax6_2_conf.out_w)
  concat_size += softmax6_2_conf.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_confidence_data_gpu + concat_size, softmax7_2_conf.output_gpu,
      softmax7_2_conf.out_c, softmax7_2_conf.out_h * softmax7_2_conf.out_w)
  concat_size += softmax7_2_conf.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_confidence_data_gpu + concat_size, softmax8_2_conf.output_gpu,
      softmax8_2_conf.out_c, softmax8_2_conf.out_h * softmax8_2_conf.out_w)
  concat_size += softmax8_2_conf.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_confidence_data_gpu + concat_size, softmax9_2_conf.output_gpu,
      softmax9_2_conf.out_c, softmax9_2_conf.out_h * softmax9_2_conf.out_w)
  concat_size += softmax9_2_conf.output_size / sizeof(float);

  transpose_chw_to_hwc_gpu(concat_confidence_data_gpu + concat_size, softmax10_2_conf.output_gpu,
      softmax10_2_conf.out_c, softmax10_2_conf.out_h * softmax10_2_conf.out_w)
  concat_size += softmax10_2_conf.output_size / sizeof(float);

  // Output is HWP from HWPC
  concat_size = 0;
  extract_max_softmax(concat_final_confidence_data_gpu,
                      concat_confidence_data_gpu,
                      4, class_num, conv4_3_norm_conf.out_h * conv4_3_norm_conf.out_c); // TODO   HWP HWP4
  concat_size += conv4_3_norm_conf.output_size / sizeof(float);

  extract_max_softmax(concat_final_confidence_data_gpu + concat_size / class_num,
                      concat_confidence_data_gpu + concat_size,
                      6, class_num, softmax_fc7_conf.out_h * softmax_fc7_conf.out_c);
  concat_size += softmax_fc7_conf.output_size / sizeof(float);

  extract_max_softmax(concat_final_confidence_data_gpu + concat_size / class_num,
                      concat_confidence_data_gpu + concat_size,
                      6, class_num, softmax6_2_conf.out_h * softmax6_2_conf.out_c);
  concat_size += softmax6_2_conf.output_size / sizeof(float);

  extract_max_softmax(concat_final_confidence_data_gpu + concat_size / class_num,
                      concat_confidence_data_gpu + concat_size,
                      6, class_num, softmax7_2_conf.out_h * softmax7_2_conf.out_c); // TODO   HWP HWP4
  concat_size += softmax7_2_conf.output_size / sizeof(float);

  extract_max_softmax(concat_final_confidence_data_gpu + concat_size / class_num
                      concat_confidence_data_gpu + concat_size,
                      6, class_num, softmax8_2_conf.out_h * softmax8_2_conf.out_c);
  concat_size += softmax8_2_conf.output_size / sizeof(float);

  extract_max_softmax(concat_final_confidence_data_gpu + concat_size / class_num,
                      concat_confidence_data_gpu + concat_size,
                      4, class_num, softmax9_2_conf.out_h * softmax9_2_conf.out_c);
  concat_size += softmax9_2_conf.output_size / sizeof(float);

  extract_max_softmax(concat_final_confidence_data_gpu + concat_size / class_num,
                      concat_confidence_data_gpu + concat_size,
                      4, class_num, softmax10_2_conf.out_h * softmax10_2_conf.out_c);
  concat_size += softmax10_2_conf.output_size / sizeof(float);

  // Output is HWP4
  multibox_decoder_gpu(concat_location_data_gpu, concat_prior_data_gpu, concat_location_data_gpu,
                       all_concat_confidence_size / 4, 0.1, 0.2);

  // TODO: Use concat_final_confidence_data_gpu, concat_location_data_gpu
  sort_box_by_confidence(concat_final_confidence_data_gpu, concat_location_data_gpu);
  
  // TODO: Sort using thrust::sort_by_key
  // indexを[0, H * W * P]までsortする。


  // Concatenate output of softmax and location and priorbox like HWP4, HWPC
  // Softmax_conf から、HWを抽出する。 元々のサイズは、HWPClass(だと思う)
  // 理由として、Reshapeした後に、shapeが[0, H*W*P, 21]となっているため
  // Locationから、HW4を抽出する。元々のサイズは、HWP（だと思う）
  // SoftMaxから抽出するとき、選択するPriorBoxも同時に抽出が必要であり、また、
  // 同時に、Location Dataの抽出も必要となります。

  // /////////////////////////////////////////////////////////
  // ///////////////         Forward        //////////////////
  // /////////////////////////////////////////////////////////
  //
  // conv_1_1.forward_gpu(conv_1_1, input_gpu);
  // gpu_to_cpu(conv_1_1.output_gpu, output, conv_1_1.output_size);
  // free_conv_layer_gpu(conv_1_1);

  // softmax_layer sl = make_softmax_layer_gpu(batch, out_c, out_h, out_w);
  // sl.forward_gpu(sl, input_gpu);
  // gpu_to_cpu(sl.output_gpu, sl.output, sl.output_size);
  // free_softmax_layer_gpu(sl);

  gpu_to_cpu(input_gpu, input, size);

  cudaFree(input_gpu);

  for (i=0; i < 4; ++i) {
    printf("%f\n", output[i]);
  }

  free(input);
  free(output);

  cudaDeviceReset();

  return 0;
}
