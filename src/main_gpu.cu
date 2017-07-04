#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
// #include <opencv/cv.h>
// #include "opencv2/opencv.hpp"
// #include "opencv2/core/core.hpp"
// #include "opencv2/gpu/gpu.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include <chrono>
// #include <sys/time.h>
// using namespace cv;
// using namespace cv::gpu;
using namespace std;
// #include <opencv2/highgui.h>


__global__ void transpose_kernel(float *d_dst, unsigned char *d_src, int w, int h)
{
  const int sx = blockDim.x * blockIdx.x + threadIdx.x;
  const int sy = blockDim.y * blockIdx.y + threadIdx.y;
  const int out_x = sx * h + sy;
  const int in_x = sy * w * 3 + sx * 3;

  if(w <= sx || h <= sy || h * w <= out_x) return;

  d_dst[out_x] = d_src[in_x] - 102.9801; // B 102.9801
  d_dst[out_x + w * h] = d_src[in_x + 1] - 115.9465; // G 115.9465
  d_dst[out_x + 2 * w * h] = d_src[in_x + 2] - 122.7717; // R 122.7717

}

void mytranspose(float *d_dst, unsigned char *d_src, int w, int h) {
  unsigned int x, y;
  unsigned int a;
  a = 16;
  x = (w - 1) / a + 1;
  y = (h - 1) / a + 1;
  dim3 d = {x, y, 1};
  dim3 d_a = {a, a, 1};
  transpose_kernel<<<d, d_a>>>(d_dst, d_src, w, h);
}
