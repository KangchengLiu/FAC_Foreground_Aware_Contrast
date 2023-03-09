#ifndef _KNNQUERYCLUSTER_CUDA_KERNEL
#define _KNNQUERYCLUSTER_CUDA_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void knnquerycluster_cuda(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor xyz_idx_tensor, at::Tensor new_xyz_tensor, at::Tensor idx_tensor, at::Tensor idx_abs_tensor, at::Tensor dist2_tensor);     // add xyz_idx_tensor

#ifdef __cplusplus
extern "C" {
#endif

void knnquerycluster_cuda_launcher(int b, int n, int m, int nsample, const float *xyz, const int *xyz_idx, const float *new_xyz, int *idx, int *idx_abs, float *dist2, cudaStream_t stream);   // add *xyz_idx

#ifdef __cplusplus
}
#endif

#endif
