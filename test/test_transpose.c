// #include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"
// #include <opencv/cv.h>
// #include "opencv2/opencv.hpp"
// #include "opencv2/core/core.hpp"
// #include "opencv2/gpu/gpu.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
#include "transpose.h"
// #include "alone_net"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_util.h"
#include "softmax.h"
// using namespace cv;
// using namespace cv::gpu;
// using namespace std;

// static void cuda_status_check(cudaError_t status, const char *file, int line) {
//   cudaDeviceSynchronize();
//   if (status != cudaSuccess) {
//     fprintf(stderr, "error [%d] : %s. in File name %s at Line %d\n", status,
//             cudaGetErrorString(status), file, line);
//     cudaDeviceReset();
//     exit(EXIT_FAILURE);
//   } else {
//     fprintf(stderr, "Success [%d]. file is %s, Line is %d\n", status, file, line);
//   }
// }
//
// #define CUDA_CHECK(status) (cuda_status_check(status, __FILE__, __LINE__));

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

	unsigned char *data;
	data = (unsigned char *)malloc(h * w * 3 * sizeof(int));
  printf("w step: %d %d\n", w, step);

	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				data[i*step + j*c + k] = 1;
			}
		}
	}

	float *image_data;
	image_data = (float*)malloc(h * w * 3 * sizeof(float));
	// cv::TickMeter timer6; timer6.start();
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				image_data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
			}
		}
	}
	// timer6.stop();
	// std::cout << "convert on CPU: " << timer6.getTimeMilli() << std::endl;

	cudaSetDevice(0);
	unsigned char *data_gpu;
	float *image_data_gpu;
	int size = h * w * 3 * sizeof(unsigned char);
	CUDA_CHECK(cudaMalloc((void**)&data_gpu, h * w * 3 * sizeof(unsigned char)));
	CUDA_CHECK(cudaMalloc((void**)&image_data_gpu, h * w * 3 * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(data_gpu, data, size, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	// cv::TickMeter timer7; timer7.start();

	transpose_gpu(image_data_gpu, data_gpu, h, w);
	cudaDeviceSynchronize();

	// timer7.stop();
	// std::cout << "Convert on GPU: " << timer7.getTimeMilli() << std::endl;
	cudaMemcpy(image_data, image_data_gpu, h * w * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  printf("Output B: %f\n", image_data[w * h * 1 - 1]);
  printf("Output G: %f\n", image_data[w * h * 2 - 1]);
  printf("Output R: %f\n", image_data[w * h * 3 - 1]);
	cudaFree(image_data_gpu);
	cudaFree(data_gpu);
	free(data);
  cudaDeviceReset();

	return 0;
}
