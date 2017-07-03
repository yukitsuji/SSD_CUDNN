#include <softmax.h>

softmax_layer make_softmax_layer_gpu(int batch, int out_c, int out_h, int out_w) {
  softmax_layer sl;
  sl.type = SOFTMAX;

  sl.out_c = out_c;
  sl.out_h = out_h;
  sl.out_w = out_w;

  sl.forward_gpu = forward_softmax_gpu;

  sl.output_size = batch * out_c * out_h * out_w * sizeof(float);
  sl.output = calloc(sl.output_size / sizeof(float), sizeof(float));
  make_gpu_array(&sl.output_gpu, 0, sl.output_size);

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&sl.inputTensorDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&sl.outputTensorDesc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(sl.inputTensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            batch, out_c, out_h, out_w));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(sl.outputTensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            batch, out_c, out_h, out_w));
  CUDA_CHECK(cudaDeviceSynchronize());
  return sl;
}

void free_softmax_layer_gpu(softmax_layer sl) {
  free(sl.output);
  cudaFree(sl.output_gpu);
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(sl.inputTensorDesc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(sl.outputTensorDesc));
}
