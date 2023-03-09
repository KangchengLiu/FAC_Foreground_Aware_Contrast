#include "../cuda_utils.h"
#include "assofixp2c_weight_cuda_kernel.h"

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)

// Note: n is the number of points, m is the number of clusters
// Note: nsample <= 1000

// input: val_c (b, n, ks) idx_c (b, n, ks) cid (b, m, 1)
// output: idx (b, m, nsample)
__global__ void assofixp2c_weight_cuda_kernel(int b, int n, int m, int ks, int nsample, const float *__restrict__ val_c, const int *__restrict__ idx_c, const int *__restrict__ cid, float *__restrict__ idx) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    //new_xyz += bs_idx * m * 3 + pt_idx * 3;
    //cid += bs_idx * m * 1 + pt_idx * 1;
    cid += bs_idx * m * 1 + pt_idx * 1;
    //xyz += bs_idx * n * 3;
    val_c += bs_idx * n * ks;  // add
    idx_c += bs_idx * n * ks;  // add
    idx += bs_idx * m * nsample + pt_idx * nsample;

    //float new_x = new_xyz[0];
    //float new_y = new_xyz[1];
    //float new_z = new_xyz[2];
    int cluster_id = cid[0];

    //double* best = new double[nsample];
    //int* besti = new int[nsample];
    //float tmpi[20];
    //for(int i = 0; i < n; i++){
    //    tmpi[i] = 0.0;
    //}
    for (int i = 0; i < nsample; i++) {
        idx[i] = 1.0;
    }
    int num = 0;
    for(int k = 0; k < n && num < nsample; k++) {
        for (int j = 0; j < ks && num < nsample; j++) {
            int id = idx_c[k * ks + j]; // cluster id of i-th point
            float val = val_c[k * ks + j];
            if (id == cluster_id) {
                //tmpi[k] = val;
                idx[num] = val;
                num++;
            }
        }
    }
    //for(int i = 0; i < n; i++){
    //    idx[i] = tmpi[i];
    //}
    //delete []best;
    //delete []besti;
}


void assofixp2c_weight_cuda_launcher(int b, int n, int m, int ks, int nsample, const float *val_c, const int *idx_c, const int *cid, float *idx, cudaStream_t stream) {
    // param val_c: (B, n, ks)
    // param idx_c: (B, n, ks)
    // param cid: (B, m, 1)
    // param idx: (B, m, n)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    assofixp2c_weight_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, ks, nsample, val_c, idx_c, cid, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
