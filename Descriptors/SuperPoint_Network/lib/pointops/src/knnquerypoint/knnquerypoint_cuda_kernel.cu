#include "../cuda_utils.h"
#include "knnquerypoint_cuda_kernel.h"

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)

// Note: n is the number of points, m is the number of clusters
// Note: nsample <= 1000

// input: idx_c (b, n, ks) cid (b, m)
// output: idx (b, m, nsample)
__global__ void knnquerypoint_cuda_kernel(int b, int n, int m, int ks, int nsample, const int *__restrict__ idx_c, const int *__restrict__ cid, int *__restrict__ idx) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    //new_xyz += bs_idx * m * 3 + pt_idx * 3;
    //cid += bs_idx * m * 1 + pt_idx * 1;
    cid += bs_idx * m;
    //xyz += bs_idx * n * 3;
    idx_c += bs_idx * n * ks;  // add
    idx += bs_idx * m * nsample + pt_idx * nsample;

    //float new_x = new_xyz[0];
    //float new_y = new_xyz[1];
    //float new_z = new_xyz[2];
    int cluster_id = cid[0];

    //double* best = new double[nsample];
    //int* besti = new int[nsample];
    int tmpi[1000];
    for(int i = 0; i < nsample; i++){
        tmpi[i] = 0;
    }
    int last = 0;
    for(int k = 0; k < n && last < nsample; k++){
        for (int j = 0; j < ks && last < nsample; j++) {
            int id = idx_c[k * ks + j]; // cluster id of i-th point
            if (id == cluster_id) {
                tmpi[last] = k;
                last++;
            }
        }
    }
    for(int i = 0; i < last; i++){
        idx[i] = tmpi[i];
    }
    for(int i = last; i < nsample; i++){
        idx[i] = cluster_id;
    }
    //delete []best;
    //delete []besti;
}


void knnquerypoint_cuda_launcher(int b, int n, int m, int ks, int nsample, const int *idx_c, const int *cid, int *idx,    cudaStream_t stream) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knnquerypoint_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, ks, nsample, idx_c, cid, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
