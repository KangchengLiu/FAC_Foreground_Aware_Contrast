#include "../cuda_utils.h"
#include "knnqueryclustergt_cuda_kernel.h"

// Note: n is the clusters, m is the points
// Note: m >> n
// Note: nsample <= 200

// input: xyz (b, n, 3) xyz_idx (b, n) xyz_gt (b, n) new_xyz (b, m, 3) new_xyz_gt (b, m, 1)
// output: idx (b, m, nsample) idx_abs (b, m, nsample) idx_gt (b, m, 1) idx_gt_abs (b, m, 1) dist2 (b, m, nsample)
__global__ void knnqueryclustergt_cuda_kernel(int b, int n, int m, int nsample, const float *__restrict__ xyz, const int *__restrict__ xyz_idx, const int *__restrict__ xyz_gt, const float *__restrict__ new_xyz, const int *__restrict__ new_xyz_gt, int *__restrict__ idx, int *__restrict__ idx_abs, int *__restrict__ idx_gt, int *__restrict__ idx_gt_abs, float *__restrict__ dist2) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    xyz_idx += bs_idx * n * 1;  // add xyz_idx initial position
    xyz_gt += bs_idx * n * 1;  // add xyz_gt initial position
    new_xyz_gt += bs_idx * m * 1 + pt_idx * 1;  // add new_xyz_gt initial position
    idx += bs_idx * m * nsample + pt_idx * nsample;
    idx_abs += bs_idx * m * nsample + pt_idx * nsample;
    idx_gt += bs_idx * m * 1 + pt_idx * 1;
    idx_gt_abs += bs_idx * m * 1 + pt_idx * 1;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int now_gt = new_xyz_gt[0]; 

    //double* best = new double[nsample];
    //int* besti = new int[nsample];
    double best[200];
    int besti_abs[200];
    int besti_gt[200];
    for(int i = 0; i < nsample; i++){
        best[i] = 1e40;
        besti_abs[i] = 0;
        besti_gt[i] = 0;
    }
    for(int k = 0; k < n; k++){
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        int tmp_gt = xyz_gt[k];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        for(int j = 0; j < nsample; j++){
            if(d2 < best[j]){
                for(int i = nsample - 1; i > j; i--){
                    best[i] = best[i - 1];
                    besti_abs[i] = besti_abs[i - 1];
                    besti_gt[i] = besti_gt[i - 1];
                }
                best[j] = d2;
                besti_abs[j] = k;
                besti_gt[j] = tmp_gt;
                break;
            }
        }
    }

    int flag = 0;
    int tmp = -1;
    for(int i = 0; i < nsample; i++){
        //idx[i] = besti_abs[i];
        idx[i] = xyz_idx[besti_abs[i]];
        idx_abs[i] = besti_abs[i];
        dist2[i] = best[i];
        if (tmp == -1) {
            tmp = besti_gt[i];
        } else if (tmp != -2 && besti_gt[i] != tmp){
            tmp = -2;
        }

        if (flag == 1) continue;
        if (besti_gt[i] == now_gt) {
            flag = 1;
            idx_gt[0] = idx[i];
            //idx_gt_abs[0] = i;
        }
    }
    if (flag == 0) { // may not exist gt due to the superpoints rates
        idx_gt[0] = idx[0];
        idx_gt_abs[0] = 0;
    }
    if (tmp == -2) {
        idx_gt_abs[0] = 1;
    } else {
        idx_gt_abs[0] = 0;
    }

    //delete []best;
    //delete []besti;
}


void knnqueryclustergt_cuda_launcher(int b, int n, int m, int nsample, const float *xyz, const int *xyz_idx, const int *xyz_gt, const float *new_xyz, const int *new_xyz_gt, int *idx, int *idx_abs, int *idx_gt, int *idx_gt_abs, float *dist2, cudaStream_t stream) {  
    // param xyz: (B, n, 3)
    // param xyz_idx: (B, n)     // add
    // param xyz_gt: (B, n)     // add
    // param new_xyz: (B, m, 3)
    // param new_xyz_gt: (B, m, 1)     // add
    // param idx: (B, m, nsample)
    // param idx_abs: (B, m, nsample)
    // param idx_gt: (B, m, 1)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knnqueryclustergt_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, nsample, xyz, xyz_idx, xyz_gt, new_xyz, new_xyz_gt, idx, idx_abs, idx_gt, idx_gt_abs, dist2);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
