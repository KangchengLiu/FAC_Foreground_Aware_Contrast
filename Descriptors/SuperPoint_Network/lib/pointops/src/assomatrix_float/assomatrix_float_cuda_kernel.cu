#include "../cuda_utils.h"
#include "assomatrix_float_cuda_kernel.h"

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)

// Note: n is the number of points, m is the number of clusters
// Note: nsample <= 1000

// input: val_c (b, n, ks) idx_c (b, n, ks) cid (b, m, 1)
// output: idx (b, m, n) cnt (b, m, 1)
__global__ void assomatrix_float_cuda_kernel(int b, int n, int m, int ks, const float *__restrict__ val_c, const int *__restrict__ idx_c, const int *__restrict__ cid, float *__restrict__ idx, int *__restrict__ cnt) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    //new_xyz += bs_idx * m * 3 + pt_idx * 3;
    //cid += bs_idx * m * 1 + pt_idx * 1;
    cid += bs_idx * m * 1 + pt_idx * 1;
    //xyz += bs_idx * n * 3;
    val_c += bs_idx * n * ks;  // add
    idx_c += bs_idx * n * ks;  // add
    idx += bs_idx * m * n + pt_idx * n;
    cnt += bs_idx * m * 1 + pt_idx * 1;     // count number of points located in one superpoint

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
    int num = 0;
    for(int k = 0; k < n; k++){
        for (int j = 0; j < ks; j++) {
            int id = idx_c[k * ks + j]; // cluster id of i-th point
            float val = val_c[k * ks + j];
            if (id == cluster_id) {
                //tmpi[k] = val;
                idx[k] = val;
                num++;
            }
        }
    }
    //for(int i = 0; i < n; i++){
    //    idx[i] = tmpi[i];
    //}
    cnt[0] = num;
    //delete []best;
    //delete []besti;
}


void assomatrix_float_cuda_launcher(int b, int n, int m, int ks, const float *val_c, const int *idx_c, const int *cid, float *idx, int *cnt, cudaStream_t stream) {
    // param val_c: (B, n, ks)
    // param idx_c: (B, n, ks)
    // param cid: (B, m, 1)
    // param idx: (B, m, n)
    // param cnt: (B, m, 1)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    assomatrix_float_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, ks, val_c, idx_c, cid, idx, cnt);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
