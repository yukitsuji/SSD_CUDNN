#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include "alone_net.h"
#include "network.h"

void transpose_gpu(float *output, unsigned char *input, int w, int h);

#endif
