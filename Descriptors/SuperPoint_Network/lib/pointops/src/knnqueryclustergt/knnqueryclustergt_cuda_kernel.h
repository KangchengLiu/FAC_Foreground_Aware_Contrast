#ifndef _KNNQUERYCLUSTERGT_CUDA_KERNEL
#define _KNNQUERYCLUSTERGT_CUDA_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void knnqueryclustergt_cuda(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor xyz_idx_tensor, at::Tensor xyz_gt_tensor, at::Tensor new_xyz_tensor, at::Tensor new_xyz_gt_tensor, at::Tensor idx_tensor, at::Tensor idx_abs_tensor, at::Tensor idx_gt_tensor, at::Tensor idx_gt_abs_tensor, at::Tensor dist2_tensor);     // add xyz_idx_tensor

#ifdef __cplusplus
extern "C" {
#endif

void knnqueryclustergt_cuda_launcher(int b, int n, int m, int nsample, const float *xyz, const int *xyz_idx, const int *xyz_gt, const float *new_xyz, const int *new_xyz_gt, int *idx, int *idx_abs, int *idx_gt, int *idx_gt_abs, float *dist2, cudaStream_t stream);   // add *xyz_idx

#ifdef __cplusplus
}
#endif

#endif
