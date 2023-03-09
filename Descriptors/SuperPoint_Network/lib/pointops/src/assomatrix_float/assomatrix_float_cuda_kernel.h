#ifndef _ASSOMATRIX_FLOAT_CUDA_KERNEL
#define _ASSOMATRIX_FLOAT_CUDA_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void assomatrix_float_cuda(int b, int n, int m, int ks, at::Tensor val_c_tensor, at::Tensor idx_c_tensor, at::Tensor cid_tensor, at::Tensor idx_tensor, at::Tensor cnt_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void assomatrix_float_cuda_launcher(int b, int n, int m, int ks, const float *val_c, const int *idx_c, const int *cid, float *idx, int *cnt, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
