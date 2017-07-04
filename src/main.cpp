#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"
#include <stdio.h>
#include <opencv/cv.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <chrono>
#include <sys/time.h>
#include "main.h"
using namespace cv;
using namespace cv::gpu;
using namespace std;
// #include <opencv2/highgui.h>

static void cuda_status_check(cudaError_t status, const char *file, int line) {
  cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    fprintf(stderr, "error [%d] : %s. in File name %s at Line %d\n", status,
            cudaGetErrorString(status), file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  } else {
    fprintf(stderr, "Success [%d]. file is %s, Line is %d\n", status, file, line);
  }
}



#define CUDA_CHECK(status) (cuda_status_check(status, __FILE__, __LINE__));

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main (int argc, char *argv[]){
	if (argc < 2){
		std::cout << "入力画像がありません。" <<std::endl;
		std::cout << "usage :" << argv[0] <<" <imagefile> " << std::endl;
		return 1;
	}
	// cv::gpu::setDevice(0);

	// cv::gpu::Stream stream;
	// cv::Mat image, image1, image2, output; //画像を入れるクラスです。実態は一次元配列です。
	// image2 = cv::imread(argv[1]);
	//
	// std::cout << image2.type() << "type \n";
	// std::cout << image1.type() << "type \n";
	//
	// cv::resize(image2, image1, Size(), 0.4, 1, cv::INTER_NEAREST);
	//
	// std::cout << image1.type() << "type \n";
	// double iStart;
	// iStart = cpuSecond();
	// GpuMat input_gpu;
	// auto start = chrono::steady_clock::now();
	// // GpuMat input_gpu (image1);
	// stream.enqueueUpload(image1, input_gpu);
	//
  // GpuMat output_gpu;
	// // output_gpu.download(output);
	// // stream.enqueueDownload(output_gpu, output);
	// output_gpu.release();
	// input_gpu.release();
	// stream.waitForCompletion();
	// // std::cout << cpuSecond() - iStart << std::endl;
	// std::cout << "Width Height : " << image1.size().width << image1.size().height << std::endl;
	//
	// std::cout << "Upload Timer: " << chrono::duration <double, milli>(chrono::steady_clock::now() - start).count() << std::endl;
	//
	// GpuMat input_gpu5;
	// stream.enqueueUpload(image1, input_gpu5);
	// stream.waitForCompletion();
	// start = chrono::steady_clock::now();
	// GpuMat output_gpu5;
	// cv::TickMeter timer9; timer9.start();
	// cv::gpu::resize(input_gpu, output_gpu5, Size(), 0.3, 0.6, cv::INTER_LINEAR);
	// output_gpu5.release();
	// input_gpu5.release();
	// stream.waitForCompletion();
	// std::cout << "Resize GPU Timer: " << chrono::duration <double, milli>(chrono::steady_clock::now() - start).count() << std::endl;
	// timer9.stop();
	//
	//
	// iStart = cpuSecond();
	// cv::TickMeter timer8; timer8.start();
	// cv::resize(image1, image, Size(), 0.3, 0.6, cv::INTER_LINEAR);
	// timer8.stop();
	// std::cout << "Resize: " << timer8.getTimeMilli() << std::endl;
	//
	// // cv::resize(image1, image, Size(), 0.5, 0.5, cv::INTER_NEAREST);
	// // std::cout << cpuSecond() - iStart << std::endl;
	//
	// iStart = cpuSecond();
	// start = chrono::steady_clock::now();
	// // GpuMat input_gpu2 (image);
	// GpuMat input_gpu2;
	// stream.enqueueUpload(image, input_gpu2);
	// GpuMat output_gpu2;
	// //auto start = chrono::steady_clock::now();
	// // output_gpu2.download(output);
	// // stream.enqueueDownload(output_gpu2, output);
	// output_gpu2.release();
	// input_gpu2.release();
	// stream.waitForCompletion();
	//
	// std::cout << "Upload Time: " << chrono::duration <double, milli>(chrono::steady_clock::now() - start).count() << std::endl;
	// std::cout << "Width Height : " << image.size().width << image.size().height << std::endl;
	//
  // cv::imshow("image",image); // "image"というウィンドウに 先述の画像用クラスを表示します。
  // cv::waitKey(0);
	// cudaDeviceReset();


	int h = 500;
	int w = 500;
  int c = 3;
	int step = w * 3;
	int i, k, j;

	unsigned char *data;
	data = (unsigned char *)malloc(h * w * 3 * sizeof(int));
	std::cout << "w step: " << w << " " << step << std::endl;
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				data[i*step + j*c + k] = 1;
			}
		}
	}

	printf("%d\n", sizeof(data[0]));

	std::cout << "ok \n";

	float *image_data;
	image_data = (float*)malloc(h * w * 3 * sizeof(float));
	cv::TickMeter timer6; timer6.start();
	std::cout << "w step: " << w << " " << step << std::endl;
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				image_data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
			}
		}
	}
	timer6.stop();
	std::cout << "convert: " << timer6.getTimeMilli() << std::endl;

	cudaSetDevice(0);
	unsigned char *data_gpu;
	float *image_data_gpu;
	int size = h * w * 3 * sizeof(unsigned char);
	CUDA_CHECK(cudaMalloc((void**)&data_gpu, h * w * 3 * sizeof(unsigned char)));
	CUDA_CHECK(cudaMalloc((void**)&image_data_gpu, h * w * 3 * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(data_gpu, data, size, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	cv::TickMeter timer7; timer7.start();

	mytranspose(image_data_gpu, data_gpu, h, w);
	cudaDeviceSynchronize();

	timer7.stop();
	std::cout << "convert: " << timer7.getTimeMilli() << std::endl;
	cudaMemcpy(image_data, image_data_gpu, h * w * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// for (i=0; i< 5000; i++) {
	// 	std::cout << image_data[i] << std::endl;
	// }

	std::cout << "OK\n";
	std::cout << "Output: " << image_data[w * h * 1 - 1] << std::endl;
	std::cout << "Output: " << image_data[w * h * 2 - 2] << std::endl;
	std::cout << "Output: " << image_data[w * h * 3 - 3] << std::endl;
	cudaFree(image_data_gpu);
	cudaFree(data_gpu);
	free(data);

	return 0;
}
