#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include "convolution.h"
#include "softmax.h"


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
  // conv_1_1 relu
  conv_layer conv1_1 = make_conv_layer_gpu(1, 2, 0, batch, 64, 3, 512, 512,
                                         3, 3, 1, 1,
                                         1, 1, 1, 1);

  conv_layer conv1_2 = make_conv_layer_gpu(1, 2, 0, batch, 64,
                                        conv1_1.out_c, conv1_1.out_h, conv1_1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  pool_layer pool1 = make_pool_layer_gpu(); // TODO

  conv_layer conv2_1 = make_conv_layer_gpu(1, 2, 0, batch, 128,
                                        pool1.out_c, pool1.out_h, pool1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  conv_layer conv2_2 = make_conv_layer_gpu(1, 2, 0, batch, 128,
                                        conv2_1.out_c, conv2_1.out_h, conv2_1.out_w,
                                        3, 3, 1, 1,
                                        1, 1, 1, 1);
  pool_layer pool2 = make_pool_layer_gpu(); // TODO

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
  pool_layer pool3 = make_pool_layer_gpu(); // TODO

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
  pool_layer pool4 = make_pool_layer_gpu(); // TODO

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
  pool_layer pool5 = make_pool_layer_gpu(); // TODO

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

  normalize_layer norm4_3 = make_norm_layer_gpu(); //TODO

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

  conv_1_1.forward_gpu(conv_1_1, input_gpu);
  gpu_to_cpu(conv_1_1.output_gpu, output, conv_1_1.output_size);
  free_conv_layer_gpu(conv_1_1);

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
