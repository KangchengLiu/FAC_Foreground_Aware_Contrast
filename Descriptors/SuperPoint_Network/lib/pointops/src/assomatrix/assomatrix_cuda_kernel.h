#ifndef _ASSOMATRIX_CUDA_KERNEL
#define _ASSOMATRIX_CUDA_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void assomatrix_cuda(int b, int n, int m, int ks, at::Tensor idx_c_tensor, at::Tensor cid_tensor, at::Tensor idx_tensor, at::Tensor cnt_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void assomatrix_cuda_launcher(int b, int n, int m, int ks, const int *idx_c, const int *cid, int *idx, int *cnt, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
