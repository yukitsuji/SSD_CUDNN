#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "element.h"
#include "cuda.h"
}

void element_max_gpu(int *output, float *input, int size, int incx) {
  CUBLAS_CHECK(cublasIsamax(cublas_handler(), size, input, incx, output));
  // transpose_kernel<<<block, grid>>>(output, input, w, h);
}
