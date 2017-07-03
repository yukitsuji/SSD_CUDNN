#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include "convolution.h"
#include "softmax.h"


int main(void){
  printf("Compile Success\n");

  // float* data = (float*)malloc(16);
  float* data = calloc(16, sizeof(float));
  float* output = calloc(16, sizeof(float));
  float* added = calloc(16, sizeof(float));
  printf("Size of data is %ld. this is for pointer size\n", sizeof(data));
  printf("Size of float is %ld\n", sizeof(float));

  int i;
  for (i=0; i < 16; ++i) {
    data[i] = (float)i;
    // printf("%f\n", data[i]);
  }

  CUDA_CHECK(cudaSetDevice(1));
  float *data_gpu;
  float *output_gpu;
  float *added_gpu;

  size_t size = sizeof(float) * 16;
  CUDA_CHECK(cudaMalloc((void**)&data_gpu, size));
  CUDA_CHECK(cudaMalloc((void**)&output_gpu, size));
  CUDA_CHECK(cudaMalloc((void**)&added_gpu, size));

  CUDA_CHECK(cudaMemcpy(data_gpu, data, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(output_gpu, output, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(added_gpu, added, size, cudaMemcpyHostToDevice));

  // softmax_layer()
  cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&srcTensorDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&dstTensorDesc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(srcTensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            2, 2, 2, 2));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(dstTensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            2, 2, 2, 2));
  CUDA_CHECK(cudaDeviceSynchronize());

  float alpha = 1;
  float beta = 1;
  cudnnHandle_t cudnnHandle;
  cudnnCreate(&cudnnHandle); // cudnn_handler()
  CUDNN_CHECK(cudnnSoftmaxForward(cudnnHandle,
                      CUDNN_SOFTMAX_FAST,
                      CUDNN_SOFTMAX_MODE_CHANNEL,
                      &alpha,
                      srcTensorDesc,
                      data_gpu,
                      &beta,
                      dstTensorDesc,
                      output_gpu));

  CUDA_CHECK(cudaDeviceSynchronize());

  add_sample(data_gpu, added_gpu);

  CUDA_CHECK(cudaMemcpy(data, data_gpu, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(output, output_gpu, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(added, added_gpu, size, cudaMemcpyDeviceToHost));

  cudaFree(data_gpu);
  cudaFree(output_gpu);
  cudaFree(added_gpu);
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(srcTensorDesc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(dstTensorDesc));

  // for (i=0; i < 16; ++i) {
  //   printf("%f\n", data[i]);
  // }

  for (i=0; i < 16; ++i) {
    printf("%f\n", added[i]);
  }

  for (i=0; i < 16; ++i) {
    printf("%f\n", output[i]);
  }

  cudaDeviceReset();
  free(data);
  free(output);
  free(added);
  return 0;
}
