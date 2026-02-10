# coding=utf-8
import torch
from torch.autograd import Function
import openpoints.cpp.grouping.knn_api as knn_api
import numpy as np
import matplotlib.pyplot as plt


class KnnFunction(Function):
    @staticmethod
    def forward(ctx,
                query_points: torch.Tensor,
                support_points: torch.Tensor,
                query_features: torch.Tensor,
                support_features: torch.Tensor,
                nsample: int,
                radius: float):
        """
        query_points: (B, M, 3)
        support_points: (B, N, 3)
        query_features: (B, M, C)
        support_features: (B, N, C)
        returns:
            group_indices: (B, M, nsample) (torch.long)
            group_values: (B, M, nsample) (torch.float)
        """
        assert query_points.is_cuda and support_points.is_cuda, "Inputs must be CUDA tensors"
        assert query_features.is_cuda and support_features.is_cuda, "Feature tensors must be CUDA"
        assert query_points.is_contiguous(), "Inputs must be contiguous"
        assert support_points.is_contiguous(), "Inputs must be contiguous"
        assert query_features.is_contiguous(), "Inputs must be contiguous"
        assert support_features.is_contiguous(), "Inputs must be contiguous"
        assert nsample == 16 or nsample == 32, "nsample must be 16 or 32"
        assert support_features.shape[2] in [16, 32, 64, 128, 256, 512,
                                             1024], "features channel must be in [16, 32, 64, 128, 256, 512, 1024]"

        B, M, _ = query_points.shape
        _, N, _ = support_points.shape

        group_indices = torch.empty((B, M, nsample), dtype=torch.long, device=query_points.device)
        group_values = torch.empty((B, M, nsample), dtype=torch.float, device=query_points.device)

        # call compiled extension
        knn_api.knn_cuda_wrapper(
            query_points,
            support_points,
            query_features,
            support_features,
            group_indices,
            group_values,
            nsample,
            radius
        )

        return group_indices, group_values

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 6


knn_func = KnnFunction.apply


class DynamicGraphTopk(Function):
    @staticmethod
    def forward(ctx,
                support_points: torch.Tensor,
                support_features: torch.Tensor,
                down_idx: torch.Tensor,
                nsample: int,
                radius: float):
        features_dist = torch.cdist(support_features, support_features)
        points_dist = torch.cdist(support_points, support_points)
        features_dist[points_dist > radius] = -1

        B, N, _ = features_dist.shape
        _, M = down_idx.shape
        K = nsample

        out_idx = torch.empty((B, M, K), dtype=torch.int64, device=support_points.device)
        out_val = torch.empty((B, M, K), dtype=torch.float32, device=support_points.device)

        # call compiled extension
        knn_api.dynamic_graph_topk_wrapper(
            features_dist,
            down_idx,
            out_idx,
            out_val,
            nsample
        )

        # values, counts = torch.unique(out_idx, return_counts=True)
        # counts = counts.cpu().numpy()
        # plt.plot(counts)
        # plt.show()
        # if support_features.shape[1] == 1024:
        #     features_dist = torch.cdist(support_features, support_features)
        #     points_dist = torch.cdist(support_points, support_points)
        #     features_dist[points_dist > radius] = -1
        #     for b in range(8):
        #         for i in range(down_idx.shape[1]):
        #             attn = np.zeros((2, support_features.shape[1]))
        #             point_a = down_idx[b][i]
        #             points_b = out_idx[b][i]
        #             attn[0] = features_dist[b][point_a].detach().cpu().numpy()
        #             attn[1] = -1
        #             if len(set(points_b.detach().cpu().numpy().tolist())) == nsample:
        #                 for point_b in points_b:
        #                     attn[1][point_b.item()] = 0
        #                 # ii = np.argsort(attn[0])
        #                 plt.plot(attn[0])
        #                 plt.plot(attn[1])
        #                 plt.show()

        return out_idx

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 6


dynamic_graph_topk = DynamicGraphTopk.apply