#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

void mytranspose(float *d_dst, unsigned char *d_src, int w, int h);
