#include <l2_normalize.h>

normalize_layer make_normalize_layer_gpu(int batch, int in_c, int in_h, int in_w) {
  normalize_layer nl;
  nl.type = NORMALIZE;

  nl.out_c = in_c;
  nl.out_h = in_h;
  nl.out_w = in_w;

  nl.forward_gpu = forward_normalize_gpu;

  nl.output_size = batch * nl.out_c * nl.out_h * nl.out_w * sizeof(float);
  nl.output = calloc(nl.output_size / sizeof(float), sizeof(float));
  make_gpu_array(&nl.output_gpu, 0, nl.output_size);

  thrust::device_vector<float> d_row_sums_4(Nrows);
  thrust::device_vector<float> d_ones(Ncols, 1.f);

  float alpha = 1.f;
  float beta  = 0.f;
  //C(m,n) = A(m,k) * B(k,n)
  //C(m,n) = A(m,k) * B(k,n)
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  CUBLAS_CHECK(cublasSgemv(cudnn_handler(), CUBLAS_OP_T, Ncols, Nrows, &alpha,
                           thrust::raw_pointer_cast(d_matrix.data()), Ncols,
                           thrust::raw_pointer_cast(d_ones.data()), 1, &beta,
                           thrust::raw_pointer_cast(d_row_sums_4.data()), 1));

  CUDA_CHECK(cudaDeviceSynchronize());
  return nl;
}

void free_normalize_layer_gpu(normalize_layer nl) {
  free(nl.output);
  cudaFree(nl.output_gpu);

}
