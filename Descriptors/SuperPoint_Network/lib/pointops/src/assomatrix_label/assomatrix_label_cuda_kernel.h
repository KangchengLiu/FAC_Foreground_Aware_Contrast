#ifndef _ASSOMATRIX_LABEL_CUDA_KERNEL
#define _ASSOMATRIX_LABEL_CUDA_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void assomatrix_label_cuda(int b, int n, int m, int ks, int category, at::Tensor idx_c_tensor, at::Tensor lab_tensor, at::Tensor cid_tensor, at::Tensor idx_tensor, at::Tensor cnt_tensor, at::Tensor clab_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void assomatrix_label_cuda_launcher(int b, int n, int m, int ks, int category, const int *idx_c, const int *lab, const int *cid, int *idx, int *cnt, int *clab, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
