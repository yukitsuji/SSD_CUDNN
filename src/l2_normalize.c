#include <l2_normalize.h>

normalize_layer make_normalize_layer_gpu(int batch, int in_c, int in_h, int in_w,
                                         int scale) {
  normalize_layer nl;
  nl.type = NORMALIZE;

  nl.out_c = in_c;
  nl.out_h = in_h;
  nl.out_w = in_w;

  nl.forward_gpu = forward_normalize_gpu;

  nl.output_size = batch * nl.out_c * nl.out_h * nl.out_w * sizeof(float);
  nl.output = calloc(nl.output_size / sizeof(float), sizeof(float));
  make_gpu_array(&nl.output_gpu, 0, nl.output_size);

  CUDA_CHECK(cudaDeviceSynchronize());

  nl.powed_output = calloc(nl.output_size / sizeof(float), sizeof(float));
  make_gpu_array(&nl.powed_output_gpu, 0, nl.output_size);

  CUDA_CHECK(cudaDeviceSynchronize());

  nl.out_norm_size = batch * 1 * nl.out_h * nl.out_w * sizeof(float);
  nl.out_norm = calloc(nl.out_norm_size / sizeof(float), sizeof(float));
  make_gpu_array(&nl.out_norm_gpu, 0, nl.out_norm_size);

  CUDA_CHECK(cudaDeviceSynchronize());

  nl.ones_channel_size = nl.out_c * sizeof(float);
  nl.ones_channel = calloc(nl.ones_channel_size / sizeof(float), sizeof(float));
  int i;
  for (i=0; i<nl.ones_channel_size / sizeof(float); ++i) nl.ones_channel[i] = 1.0f;
  make_gpu_array(&nl.ones_channel_gpu, nl.ones_channel, nl.ones_channel_size);

  CUDA_CHECK(cudaDeviceSynchronize());

  nl.scale_size = nl.out_c * sizeof(float);
  nl.scale = calloc(nl.scale_size / sizeof(float), sizeof(float));
  for (i=0; i<nl.ones_channel_size / sizeof(float); ++i) nl.scale[i] = i;
  make_gpu_array(&nl.scale_gpu, nl.scale, nl.scale_size);

  CUDA_CHECK(cudaDeviceSynchronize());
  return nl;
}

void free_normalize_layer_gpu(normalize_layer nl) {
  free(nl.output);
  cudaFree(nl.output_gpu);
  free(nl.powed_output);
  cudaFree(nl.powed_output_gpu);
  free(nl.out_norm);
  cudaFree(nl.out_norm_gpu);
  free(nl.ones_channel);
  cudaFree(nl.ones_channel_gpu);

  free(nl.scale);
  cudaFree(nl.scale_gpu);
}
