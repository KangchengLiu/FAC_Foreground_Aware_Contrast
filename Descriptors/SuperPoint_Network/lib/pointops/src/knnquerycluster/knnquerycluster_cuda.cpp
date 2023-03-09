#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "knnquerycluster_cuda_kernel.h"

extern THCState *state;

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


// add xyz_idx_tensor
void knnquerycluster_cuda(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor xyz_idx_tensor, at::Tensor new_xyz_tensor, at::Tensor idx_tensor, at::Tensor idx_abs_tensor, at::Tensor dist2_tensor)
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(xyz_idx_tensor);    // add CHECK_INPUT of xyz_idx_tensor

    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *xyz_idx = xyz_idx_tensor.data<int>();  // add *xyz_idx
    int *idx = idx_tensor.data<int>();
    int *idx_abs = idx_abs_tensor.data<int>();
    float *dist2 = dist2_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);

    knnquerycluster_cuda_launcher(b, n, m, nsample, xyz, xyz_idx, new_xyz, idx, idx_abs, dist2, stream); // add xyz_idx
}
