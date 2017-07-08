#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include "alone_net.h"
#include "network.h"

void element_max_gpu(int *output, float *input, int size, int incx);

#endif
