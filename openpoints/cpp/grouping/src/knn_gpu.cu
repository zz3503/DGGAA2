#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <inttypes.h>
#include <cassert>

// ---------------- config ----------------
#define BLOCK_SIZE 128
#define MAX_K     32
#define MAX_N     2048
#define MAX_C     1024   // feature dimension upper bound

// ---------------- device constant ----------------
__constant__ float d_radius2;
// __constant__ float LOG_TABLE[32] = {
//     1.000000,1.041393,1.079181,1.113943,1.146128,1.176091,1.204120,1.230449,
//     1.255273,1.278754,1.301030,1.322219,1.342423,1.361728,1.380211,1.397940,
//     1.414973,1.431364,1.447158,1.462398,1.477121,1.491362,1.505150,1.518514,
//     1.531479,1.544068,1.556303,1.568202,1.579784,1.591065,1.602060,1.612784
// };

__constant__ float LOG_TABLE[32] = {
    1.000000,1.313262,1.551445,1.743668,1.904832,2.043592,2.165422,2.274009,
    2.371951,2.461150,2.543040,2.618729,2.689090,2.754824,2.816503,2.874597,
    2.929501,2.981546,3.031016,3.078154,3.123170,3.166246,3.207543,3.247202,
    3.285348,3.322092,3.357534,3.391762,3.424858,3.456893,3.487934,3.518040
};

// ---------------- error check ----------------
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// -------------- small insertion sort to keep K smallest --------------
template<int K>
__device__ void bitonicK(float* val, int* idx)
{
#pragma unroll
    for (int i = 1; i < K; ++i)
    {
        float v = val[i];
        int   id = idx[i];
#pragma unroll
        for (int j = i; j > 0 && val[j - 1] > v; --j)
        {
            val[j] = val[j - 1]; idx[j] = idx[j - 1];
            val[j - 1] = v;      idx[j - 1] = id;
        }
    }
}

template<int K>
__device__ void bitonicK(float* val, int* idx, int* deep)
{
#pragma unroll
    for (int i = 1; i < K; ++i)
    {
        float v = val[i];
        int   id = idx[i];
        int   d = deep[i];
#pragma unroll
        for (int j = i; j > 0 && val[j - 1] > v; --j)
        {
            val[j] = val[j - 1]; idx[j] = idx[j - 1]; deep[j] = deep[j - 1];
            val[j - 1] = v;      idx[j - 1] = id;     deep[j - 1] = d;
        }
    }
}

__global__ void ball_dist_kernel(
    const float* __restrict__ query_feat,   // (B, M, C)
    const float* __restrict__ support_feat, // (B, N, C)
    const float* __restrict__ query_pts,    // (B, M, 3)
    const float* __restrict__ support_pts,  // (B, N, 3)
    float* __restrict__ dist,               // 输出 (B, M, N)
    int B, int M, int N, int C)
{
    int b = blockIdx.z;
    int m = blockIdx.y * blockDim.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || m >= M || n >= N) return;

    // 计算点距离
    float qx = query_pts[((b * M + m) * 3) + 0];
    float qy = query_pts[((b * M + m) * 3) + 1];
    float qz = query_pts[((b * M + m) * 3) + 2];

    float sx = support_pts[((b * N + n) * 3) + 0];
    float sy = support_pts[((b * N + n) * 3) + 1];
    float sz = support_pts[((b * N + n) * 3) + 2];

    float dx = qx - sx;
    float dy = qy - sy;
    float dz = qz - sz;
    float d2 = dx * dx + dy * dy + dz * dz;

    if (d2>d_radius2)
    {
        dist[((b * M + m) * N) + n] = -1.0f;
        return;
    }
    // 计算特征距离
    float feat_dist = 0.0f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float qf = query_feat[((b * M + m) * C) + c];
        float sf = support_feat[((b * N + n) * C) + c];
        float df = qf - sf;
        feat_dist += df * df;
    }

    dist[((b * M + m) * N) + n] = feat_dist;
}

void ball_dist_launcher(
    const float* __restrict__ query_feat,   // (B, M, C)
    const float* __restrict__ support_feat, // (B, N, C)
    const float* __restrict__ query_pts,    // (B, M, 3)
    const float* __restrict__ support_pts,  // (B, N, 3)
    float* __restrict__ dist,               // 输出 (B, M, N)
    int B, int M, int N, int C, float radius)
{
    float radius2_host = radius * radius;
    CUDA_CHECK(cudaMemcpyToSymbol(d_radius2, &radius2_host, sizeof(float)));

    dim3 blocks(BLOCK_SIZE);
    dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, M, B);

    ball_dist_kernel<< <grid, blocks >> > (query_feat, support_feat, query_pts, support_pts, dist, B, M, N, C);

    CUDA_CHECK(cudaGetLastError());
}


// -------------- main kernel --------------
template<int K, int C>
__global__ void knn_kernel(
    const float* __restrict__ q_pts,   // B*M*3
    const float* __restrict__ s_pts,   // B*N*3
    const float* __restrict__ q_feat,  // B*M*C
    const float* __restrict__ s_feat,  // B*N*C
    int64_t* out_idx,                  // B*M*K
    float* out_val,                  // B*M*K
    int B, int M, int N)
{
    const int b = blockIdx.y;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || m >= M) return;

    // query point and feature
    float qx = q_pts[(b * M + m) * 3 + 0];
    float qy = q_pts[(b * M + m) * 3 + 1];
    float qz = q_pts[(b * M + m) * 3 + 2];

    float qf[MAX_C];
#pragma unroll
    for (int c = 0; c < C; ++c)
        qf[c] = q_feat[(b * M + m) * C + c];

    float val[K];
    int   idx[K];
#pragma unroll
    for (int i = 0; i < K; ++i) { val[i] = FLT_MAX; idx[i] = -1; }

    // fallback 全局最小
    float min_val = FLT_MAX;
    int   min_idx = -1;

    extern __shared__ float shm[];
    float* tile_pts = shm;

    for (int n_tile = 0; n_tile < N; n_tile += MAX_N)
    {
        for (int i = threadIdx.x; i < MAX_N * 3; i += blockDim.x)
        {
            int off = n_tile * 3 + i;
            tile_pts[i] = (off < N * 3) ? s_pts[b * N * 3 + off] : 0.0f;
        }
        __syncthreads();

        int tile_len = min(MAX_N, N - n_tile);
        for (int i = 0; i < tile_len; ++i)
        {
            int n = n_tile + i;
            float sx = tile_pts[i * 3 + 0];
            float sy = tile_pts[i * 3 + 1];
            float sz = tile_pts[i * 3 + 2];

            float dx = sx - qx, dy = sy - qy, dz = sz - qz;
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > d_radius2) continue;

            // 计算 feature L2 距离
            float df = 0.0f;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                float tmp = qf[c] - s_feat[(b * N + n) * C + c];
                df += tmp * tmp;
            }

            // 维护全局最小距离
            if (df < min_val) { min_val = df; min_idx = n; }

            // 如果在半径内，才插入 K 最小值
            if (d2 <= d_radius2) {
                if (df < val[K - 1]) {
                    val[K - 1] = df;
                    idx[K - 1] = n;
                    bitonicK<K>(val, idx);
                }
            }
        }
        __syncthreads();
    }

    // write back
    for (int k = 0; k < K; ++k) {
        out_idx[(b * M + m) * K + k] = (idx[k] < 0) ? min_idx : (int64_t)idx[k];
        out_val[(b * M + m) * K + k] = (idx[k] < 0) ? min_val : val[k];
    }
}

// ------------------------------------------------------------
// CUDA Launcher
// ------------------------------------------------------------
void knn_cuda_launcher(
    const float* query_points,
    const float* support_points,
    const float* query_features,
    const float* support_features,
    int64_t* group_indices,
    float* group_value,
    int B, int M, int N, int C, int K, float radius)
{
    float radius2_host = radius * radius;
    CUDA_CHECK(cudaMemcpyToSymbol(d_radius2, &radius2_host, sizeof(float)));

    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, B);
    size_t shared_bytes = MAX_N * 3 * sizeof(float);

    switch (K) {
        case 16:
            switch (C) {
                case 16:knn_kernel<16, 16> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 32:knn_kernel<16, 32> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 64:knn_kernel<16, 64> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 128:knn_kernel<16, 128> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 256:knn_kernel<16, 256> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 512:knn_kernel<16, 512> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 1024:knn_kernel<16, 1024> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                default:break;
            }
        break;
        case 32:
            switch (C) {
                case 16:knn_kernel<32, 16> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 32:knn_kernel<32, 32> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 64:knn_kernel<32, 64> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 128:knn_kernel<32, 128> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 256:knn_kernel<32, 256> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 512:knn_kernel<32, 512> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                case 1024:knn_kernel<32, 1024> << <grid, block, shared_bytes >> > (query_points, support_points, query_features, support_features, group_indices, group_value, B, M, N);break;
                default:break;
            }
        break;
        default:break;
    }

    CUDA_CHECK(cudaGetLastError());
}

template<int K>
__global__ void dynamic_graph_topk_kernel(
    int B, int N, int M,
    const float* __restrict__ feat_dist,  // (B,N,N)
    const int64_t* __restrict__ down_idx, // (B,M)
    int64_t* __restrict__ out_idx,         // (B,M,K)
    float* __restrict__ out_val         // (B,M,K)
){
int b = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if(b>=B || m>=M) return;

    feat_dist += b * N * N;
    down_idx  += b * M;
    out_idx   += b * M * K + m * K;
    out_val   += b * M * K + m * K;

    int q = (int)down_idx[m];

    float val[K];
    int   idx[K];
    int   deep[K];
#pragma unroll
    for(int i=0;i<K;i++){
        idx[i] = q;
        val[i] = 1000.f;
        deep[i] = 0;
    }
#pragma unroll
    for(int now=N-1;now>=0;now--){
        float qv = feat_dist[q*N + now];
        if(qv < 0 || now ==q) continue;
#pragma unroll
        for (int i=K-1;i>=0;i--){
            float temp = feat_dist[now*N + idx[i]] * LOG_TABLE[deep[i]];
            if(temp < val[K-2] && temp > 0){
                val[K-2] = temp;
                idx[K-2] = now;
                deep[K-2] = deep[i] + 1;
                bitonicK<K>(val,idx,deep);
//                 if (b == 0 && m == 0) {
//                     for (int t = 0; t < K; t++) {
//                         printf("%d %.5f,",deep[t], val[t]);
//                     }
//                     printf("\n");
//                 }
                break;
            }
        }
    }

#pragma unroll
    for(int i=0;i<K;i++)
    {
        out_idx[i] = idx[i];
        out_val[i] = val[i];
    }
}

void dynamic_graph_topk_launcher(
    int B, int N, int M, int K,
    const float* __restrict__ feat_dist,  // (B,N,N)
    const int64_t* __restrict__ down_idx, // (B,M)
    int64_t* __restrict__ out_idx,         // (B,M,K)
    float* __restrict__ out_val         // (B,M,K)
){
    dim3 blocks(BLOCK_SIZE);
    dim3 grid((M+BLOCK_SIZE-1)/BLOCK_SIZE, B);

    dynamic_graph_topk_kernel<32><< <grid, blocks >> > (B,N,M,feat_dist,down_idx,out_idx,out_val);

    CUDA_CHECK(cudaGetLastError());
}
