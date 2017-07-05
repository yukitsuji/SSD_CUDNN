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
  // int lda=m,ldb=k,ldc=m;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  cublasSgemv(Caffe::cublas_handle(), cuTransA, CHANNEL, W*H, &alpha, A, CHANNEL, x, 1, &beta, y, 1)

  CUBLAS_CHECK(cublasSgemv(cudnn_handler(), CUBLAS_OP_T, Ncols, Nrows, &alpha,
                           thrust::raw_pointer_cast(d_matrix.data()), Ncols,
                           thrust::raw_pointer_cast(d_ones.data()), 1, &beta,
                           thrust::raw_pointer_cast(d_row_sums_4.data()), 1));

   const Dtype* sum_channel_multiplier = sum_channel_multiplier_.gpu_data();
   int num = bottom[0]->num();
   int dim = bottom[0]->count() / num;
   int spatial_dim = bottom[0]->height() * bottom[0]->width();
   int channels = bottom[0]->channels();
   for (int n = 0; n < num; ++n) {
     caffe_gpu_powx<Dtype>(dim, bottom_data, Dtype(2), buffer_data);
     if (across_spatial_) {
       Dtype normsqr;
       caffe_gpu_asum<Dtype>(dim, buffer_data, &normsqr);
       // add eps to avoid overflow
       norm_data[n] = pow(normsqr+eps_, Dtype(0.5));
       caffe_gpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data,
                              top_data);
     } else {
       // compute norm
       caffe_gpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                             buffer_data, sum_channel_multiplier, Dtype(1),
                             norm_data);
       caffe_gpu_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
       // scale the layer
       // NOLINT_NEXT_LINE(whitespace/operators)
       DivBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
           dim, bottom_data, norm_data, channels, spatial_dim, CblasNoTrans,
           top_data);
       CUDA_POST_KERNEL_CHECK;
       norm_data += spatial_dim;
     }
     // scale the output
     if (channel_shared_) {
       caffe_gpu_scal<Dtype>(dim, scale[0], top_data);
     } else {
       MulBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
           dim, top_data, scale, channels, spatial_dim, CblasTrans,
           top_data);
       CUDA_POST_KERNEL_CHECK;
     }
     bottom_data += dim;
     top_data += dim;
   }

  CUDA_CHECK(cudaDeviceSynchronize());
  return nl;
}

void free_normalize_layer_gpu(normalize_layer nl) {
  free(nl.output);
  cudaFree(nl.output_gpu);

}
