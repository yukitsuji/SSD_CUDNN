#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include "convolution.h"
#include "softmax.h"
#include "pooling.h"


int main(void){
  printf("Compile Success\n");

  float* input = calloc(16, sizeof(float));
  float* output = calloc(16, sizeof(float));
  float* added = calloc(16, sizeof(float));
  float *set_cpu = calloc(16, sizeof(float));
  printf("Size of input is %ld. this is for pointer size\n", sizeof(input));
  printf("Size of float is %ld\n", sizeof(float));

  int i;
  for (i=0; i < 16; ++i) {
    input[i] = (float)i;
  }

  CUDA_CHECK(cudaSetDevice(0));
  float *input_gpu;
  float *added_gpu;
  float *set_gpu;

  size_t size = sizeof(float) * 16;
  make_gpu_array(&input_gpu, input, size);
  make_gpu_array(&added_gpu, added, size);
  make_gpu_array(&set_gpu, 0, size);

  int batch = 2;
  int out_c = 2;
  int out_h = 2;
  int out_w = 2;
  // data_layer = make_data_layer_gpu(data, batch, data_c, data_h, data_w);
  // prev_layer = make_prev_layer_gpu(data_layer, batch, out_c, out_h, out_w);
  // softmax_layer sl = make_softmax_layer_gpu(prev_layer, batch, out_c, out_h, out_w);

  // make_network_gpu(last_layer); 連結リストで順に実行される。 -> 連結リストを作成する。
  // network.predict(); network.train();
  // sl.forward();

  // conv_layer cl1 = make_conv_layer_gpu(1, 2, 0, batch, 2, 2, 2, 2,
  //                                        2, 2, 0, 0,
  //                                        1, 1, 1, 1);
  // cl1.forward_gpu(cl1, input_gpu);
  //
  // // add bias layer
  // // cl1_bias = make_bias_layer_gpu();
  // // cl1_bias.forward(cl1_bias, )
  //
  // gpu_to_cpu(cl1.output_gpu, output, cl1.output_size);
  // free_conv_layer_gpu(cl1);

  pool_layer cl1 = make_pool_layer_gpu(0,
                                       batch, 2, 2, 2,
                                         2, 2, 0, 0,
                                         1, 1);
  cl1.output_gpu += 1;
  cl1.forward_gpu(cl1, input_gpu);

  // add bias layer
  // cl1_bias = make_bias_layer_gpu();
  // cl1_bias.forward(cl1_bias, )

  gpu_to_cpu(cl1.output_gpu, output, cl1.output_size);
  free_pool_layer_gpu(cl1);

  // softmax_layer sl = make_softmax_layer_gpu(batch, out_c, out_h, out_w);
  // sl.forward_gpu(sl, input_gpu);
  // gpu_to_cpu(sl.output_gpu, sl.output, sl.output_size);
  // free_softmax_layer_gpu(sl);

  add_sample(input_gpu, added_gpu);

  gpu_to_cpu(input_gpu, input, size);
  gpu_to_cpu(added_gpu, added, size);
  gpu_to_cpu(set_gpu, set_cpu, size);

  cudaFree(input_gpu);
  cudaFree(added_gpu);
  cudaFree(set_gpu);

  // for (i=0; i < 16; ++i) {
  //   printf("%f\n", added[i]);
  // }
  //
  for (i=0; i < 16; ++i) {
    printf("%f\n", output[i]);
  }
  //
  // for (i=0; i < 16; ++i) {
  //   printf("%f\n", set_cpu[i]);
  // }
  free(input);
  free(output);
  free(added);
  free(set_cpu);

  cudaDeviceReset();

  return 0;
}
