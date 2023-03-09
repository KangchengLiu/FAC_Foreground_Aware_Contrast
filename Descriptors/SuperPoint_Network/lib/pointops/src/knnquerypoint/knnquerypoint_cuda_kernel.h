#ifndef _KNNQUERYPOINT_CUDA_KERNEL
#define _KNNQUERYPOINT_CUDA_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void knnquerypoint_cuda(int b, int n, int m, int ks, int nsample, at::Tensor idx_c_tensor, at::Tensor cid_tensor, at::Tensor idx_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void knnquerypoint_cuda_launcher(int b, int n, int m, int ks, int nsample, const int *idx_c, const int *cid, int *idx, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
