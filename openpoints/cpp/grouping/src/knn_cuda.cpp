// knn_cuda.cpp
#include "knn_cuda.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void knn_cuda_wrapper(
    at::Tensor query_points,
    at::Tensor support_points,
    at::Tensor query_features,
    at::Tensor support_features,
    at::Tensor group_indices,
    at::Tensor group_values,
    int nsample,
    float radius)
{
    // basic checks
    TORCH_CHECK(query_points.device().is_cuda(), "query_points must be CUDA tensor");
    TORCH_CHECK(support_points.device().is_cuda(), "support_points must be CUDA tensor");
    TORCH_CHECK(query_features.device().is_cuda(), "query_features must be CUDA tensor");
    TORCH_CHECK(support_features.device().is_cuda(), "support_features must be CUDA tensor");
    TORCH_CHECK(group_indices.device().is_cuda(), "group_indices must be CUDA tensor");
    TORCH_CHECK(group_values.device().is_cuda(), "group_values must be CUDA tensor");

    TORCH_CHECK(query_points.dim() == 3 && query_points.size(2) == 3, "query_points should be [B,M,3]");
    TORCH_CHECK(support_points.dim() == 3 && support_points.size(2) == 3, "support_points should be [B,N,3]");
    TORCH_CHECK(query_features.dim() == 3, "query_features should be [B,M,C]");
    TORCH_CHECK(support_features.dim() == 3, "support_features should be [B,N,C]");

    int B = (int)query_points.size(0);
    int M = (int)query_points.size(1);
    int N = (int)support_points.size(1);
    int C = (int)query_features.size(2);
    int K = nsample;

    TORCH_CHECK(group_indices.size(0) == B && group_indices.size(1) == M && group_indices.size(2) == K,
        "group_indices shape must be [B,M,K]");
    TORCH_CHECK(group_values.size(0) == B && group_values.size(1) == M && group_values.size(2) == K,
        "group_values shape must be [B,M,K]");

    TORCH_CHECK(query_points.dtype() == at::kFloat, "query_points must be float");
    TORCH_CHECK(support_points.dtype() == at::kFloat, "support_points must be float");
    TORCH_CHECK(query_features.dtype() == at::kFloat, "query_features must be float");
    TORCH_CHECK(support_features.dtype() == at::kFloat, "support_features must be float");
    TORCH_CHECK(group_indices.dtype() == at::kLong, "group_indices must be int64");
    TORCH_CHECK(group_values.dtype() == at::kFloat, "group_values must be float32");

    // ensure contiguous
    if (!query_points.is_contiguous()) query_points = query_points.contiguous();
    if (!support_points.is_contiguous()) support_points = support_points.contiguous();
    if (!query_features.is_contiguous()) query_features = query_features.contiguous();
    if (!support_features.is_contiguous()) support_features = support_features.contiguous();
    if (!group_indices.is_contiguous()) group_indices = group_indices.contiguous();
    if (!group_values.is_contiguous()) group_values = group_values.contiguous();

    const float* d_q_pts = query_points.data_ptr<float>();
    const float* d_s_pts = support_points.data_ptr<float>();
    const float* d_q_feat = query_features.data_ptr<float>();
    const float* d_s_feat = support_features.data_ptr<float>();
    int64_t* d_out_idx = group_indices.data_ptr<int64_t>();
    float* d_out_dist = group_values.data_ptr<float>();

    // call dispatcher
    knn_cuda_launcher(d_q_pts, d_s_pts, d_q_feat, d_s_feat, d_out_idx, d_out_dist, B, M, N, C, K, radius);

}

void ball_dist_wrapper(
    at::Tensor query_points,
    at::Tensor support_points,
    at::Tensor query_features,
    at::Tensor support_features,
    at::Tensor dist,
    float radius)
{
    // basic checks
    TORCH_CHECK(query_points.device().is_cuda(), "query_points must be CUDA tensor");
    TORCH_CHECK(support_points.device().is_cuda(), "support_points must be CUDA tensor");
    TORCH_CHECK(query_features.device().is_cuda(), "query_features must be CUDA tensor");
    TORCH_CHECK(support_features.device().is_cuda(), "support_features must be CUDA tensor");
    TORCH_CHECK(dist.device().is_cuda(), "group_indices must be CUDA tensor");

    TORCH_CHECK(query_points.dim() == 3 && query_points.size(2) == 3, "query_points should be [B,M,3]");
    TORCH_CHECK(support_points.dim() == 3 && support_points.size(2) == 3, "support_points should be [B,N,3]");
    TORCH_CHECK(query_features.dim() == 3, "query_features should be [B,M,C]");
    TORCH_CHECK(support_features.dim() == 3, "support_features should be [B,N,C]");

    int B = (int)query_points.size(0);
    int M = (int)query_points.size(1);
    int N = (int)support_points.size(1);
    int C = (int)query_features.size(2);

    TORCH_CHECK(dist.size(0) == B && dist.size(1) == M && dist.size(2) == N,
        "dist shape must be [B,M,N]");

    TORCH_CHECK(query_points.dtype() == at::kFloat, "query_points must be float");
    TORCH_CHECK(support_points.dtype() == at::kFloat, "support_points must be float");
    TORCH_CHECK(query_features.dtype() == at::kFloat, "query_features must be float");
    TORCH_CHECK(support_features.dtype() == at::kFloat, "support_features must be float");
    TORCH_CHECK(dist.dtype() == at::kFloat, "dist must be float32");

    // ensure contiguous
    if (!query_points.is_contiguous()) query_points = query_points.contiguous();
    if (!support_points.is_contiguous()) support_points = support_points.contiguous();
    if (!query_features.is_contiguous()) query_features = query_features.contiguous();
    if (!support_features.is_contiguous()) support_features = support_features.contiguous();
    if (!dist.is_contiguous()) dist = dist.contiguous();

    const float* d_q_pts = query_points.data_ptr<float>();
    const float* d_s_pts = support_points.data_ptr<float>();
    const float* d_q_feat = query_features.data_ptr<float>();
    const float* d_s_feat = support_features.data_ptr<float>();
    float* d_out_dist = dist.data_ptr<float>();

    // call dispatcher
    ball_dist_launcher(d_q_feat, d_s_feat, d_q_pts, d_s_pts, d_out_dist, B, M, N, C, radius);

}


void dynamic_graph_topk_wrapper(
    at::Tensor feat_dist_tensor,
    at::Tensor down_idx_tensor,
    at::Tensor out_idx_tensor,
    at::Tensor out_val_tensor,
    int nsample)
{
    // basic checks
    TORCH_CHECK(feat_dist_tensor.device().is_cuda(), "feat_dist_tensor must be CUDA tensor");
    TORCH_CHECK(down_idx_tensor.device().is_cuda(), "down_idx_tensor must be CUDA tensor");
    TORCH_CHECK(out_idx_tensor.device().is_cuda(), "out_idx_tensor must be CUDA tensor");

    int B = (int)feat_dist_tensor.size(0);
    int N = (int)feat_dist_tensor.size(1);
    int M = (int)down_idx_tensor.size(1);
    int K = nsample;

    TORCH_CHECK(feat_dist_tensor.dtype() == at::kFloat, "feat_dist_tensor must be float");
    TORCH_CHECK(down_idx_tensor.dtype() == at::kLong, "down_idx_tensor must be int64");
    TORCH_CHECK(out_idx_tensor.dtype() == at::kLong, "out_idx_tensor must be int64");

    // ensure contiguous
    if (!feat_dist_tensor.is_contiguous()) feat_dist_tensor = feat_dist_tensor.contiguous();
    if (!down_idx_tensor.is_contiguous()) down_idx_tensor = down_idx_tensor.contiguous();
    if (!out_idx_tensor.is_contiguous()) out_idx_tensor = out_idx_tensor.contiguous();

    const float* feat_dist = feat_dist_tensor.data_ptr<float>();
    const int64_t* down_idx = down_idx_tensor.data_ptr<int64_t>();
    int64_t* out_idx = out_idx_tensor.data_ptr<int64_t>();
    float* out_val = out_val_tensor.data_ptr<float>();

    // call dispatcher
    dynamic_graph_topk_launcher(B, N, M, K, feat_dist, down_idx, out_idx, out_val);

}