#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include "sampling_cuda_kernel.h"

extern THCState *state;

// --------------------------- gathering ----------------------
void gathering_forward_cuda(int b, int c, int n, int m, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{
    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();
    gathering_forward_cuda_launcher(b, c, n, m, points, idx, out);
}

void gathering_backward_cuda(int b, int c, int n, int m, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor)
{

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_points = grad_points_tensor.data<float>();
    gathering_backward_cuda_launcher(b, c, n, m, grad_out, idx, grad_points);
}

// --------------------------- gathering int ----------------------
void gathering_int_forward_cuda(int b, int c, int n, int m, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{
    const int *points = points_tensor.data<int>();
    const int *idx = idx_tensor.data<int>();
    int *out = out_tensor.data<int>();
    gathering_int_forward_cuda_launcher(b, c, n, m, points, idx, out);
}

void gathering_int_backward_cuda(int b, int c, int n, int m, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor)
{
    
    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_points = grad_points_tensor.data<float>();
    gathering_int_backward_cuda_launcher(b, c, n, m, grad_out, idx, grad_points);
}

// --------------------------- gathering cluster ----------------------
// add for finally gather cluster for each point
void gathering_cluster_forward_cuda(int b, int c, int n, int m, int k, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor idx_3d_tensor, at::Tensor out_tensor)
{
    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const int *idx_3d = idx_3d_tensor.data<int>();      // add
    float *out = out_tensor.data<float>();
    gathering_cluster_forward_cuda_launcher(b, c, n, m, k, points, idx, idx_3d, out);              // modify
}

void gathering_cluster_backward_cuda(int b, int c, int n, int m, int k, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor idx_3d_tensor, at::Tensor grad_points_tensor)
{

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const int *idx_3d = idx_3d_tensor.data<int>();      // add
    float *grad_points = grad_points_tensor.data<float>();
    gathering_cluster_backward_cuda_launcher(b, c, n, m, k, grad_out, idx, idx_3d, grad_points);   // modify
}

void furthestsampling_cuda(int b, int n, int m, at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor)
{
    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
    furthestsampling_cuda_launcher(b, n, m, points, temp, idx);
}
