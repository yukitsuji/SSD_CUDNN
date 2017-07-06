// #include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"
#include "l2_normalize.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_util.h"


int main (int argc, char *argv[]){
	if (argc < 2){
		printf("入力画像がない\n");
		return 1;
	}

	int h = 500;
	int w = 500;
  int c = 3;
	int step = w * 3;
	int i, k, j;

	float *data;
	data = calloc(h * w * 3, sizeof(float));
  printf("w step: %d %d\n", w, step);

	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				data[i*step + j*c + k] = 2;
			}
		}
	}

	float *image_data;
	image_data = calloc(h*w*3, sizeof(float));
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				image_data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
			}
		}
	}

	cudaSetDevice(0);
	float *data_gpu;
	float *image_data_gpu;
	int size = h * w * 3 * sizeof(float);

	CUDA_CHECK(cudaMalloc((void**)&data_gpu, size));
	// CUDA_CHECK(cudaMalloc((void**)&image_data_gpu, h * w * 3 * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(data_gpu, data, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaDeviceSynchronize());

  normalize_layer nl = make_normalize_layer_gpu(1, 3, h, w, 0);
	CUDA_CHECK(cudaDeviceSynchronize());
  nl.forward_gpu(nl, data_gpu);
	CUDA_CHECK(cudaDeviceSynchronize());
  gpu_to_cpu(nl.powed_output_gpu, image_data, nl.output_size);
	CUDA_CHECK(cudaDeviceSynchronize());
  free_normalize_layer_gpu(nl);

	// transpose_gpu(image_data_gpu, data_gpu, h, w);
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("Hello World\n");

  printf("Output B: %f\n", image_data[w * h * 1 - 1]);
  printf("Output G: %f\n", image_data[w * h * 2 - 1]);
  printf("Output R: %f\n", image_data[w * h * 3 - 1]);
	// cudaFree(image_data_gpu);
	cudaFree(data_gpu);
	free(data);
  cudaDeviceReset();

	return 0;
}
