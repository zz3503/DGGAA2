#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <cuda_runtime.h>

// Host-dispatcher that launches the GPU kernel (implemented in knn_gpu.cu)
void knn_cuda_launcher(
    const float* query_points,
    const float* support_points,
    const float* query_features,
    const float* support_features,
    int64_t* group_indices,
    float* group_value,
    int B, int M, int N, int C, int K, float radius);

// PyTorch wrapper (implemented in knn_cuda.cpp)
// Copies radius^2 to device constant and calls the launcher.
void knn_cuda_wrapper(
    at::Tensor query_points,    // [B,M,3] float32 CUDA
    at::Tensor support_points,  // [B,N,3] float32 CUDA
    at::Tensor query_features,  // [B,M,C] float32 CUDA
    at::Tensor support_features,// [B,N,C] float32 CUDA
    at::Tensor group_indices,   // [B,M,K] int64 CUDA (output)
    at::Tensor group_values,    // [B,M,K] float CUDA (output)
    int nsample,
    float radius);


void ball_dist_launcher(
    const float* __restrict__ query_feat,   // (B, M, C)
    const float* __restrict__ support_feat, // (B, N, C)
    const float* __restrict__ query_pts,    // (B, M, 3)
    const float* __restrict__ support_pts,  // (B, N, 3)
    float* __restrict__ dist,               // Êä³ö (B, M, N)
    int B, int M, int N, int C, float radius);

void ball_dist_wrapper(
    at::Tensor query_points,
    at::Tensor support_points,
    at::Tensor query_features,
    at::Tensor support_features,
    at::Tensor dist,
    float radius);

void dynamic_graph_topk_launcher(
    int B, int N, int M, int K,
    const float* __restrict__ feat_dist,  // (B,N,N)
    const int64_t* __restrict__ down_idx, // (B,M)
    int64_t* __restrict__ out_idx,         // (B,M,K)
    float* __restrict__ out_val         // (B,M,K)
);

void dynamic_graph_topk_wrapper(
    at::Tensor feat_dist_tensor,
    at::Tensor down_idx_tensor,
    at::Tensor out_idx_tensor,
    at::Tensor out_val_tensor,
    int nsample);
