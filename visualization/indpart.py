import os
import numpy as np
import torch
import ctypes
import pyvista as pv
import math
from scipy.optimize import linear_sum_assignment
from openpoints.models import build_model_from_cfg
from openpoints.utils import cal_model_parm_nums, EasyConfig
from openpoints.transforms.point_transformer_gpu import PointCloudCenterAndNormalize, PointCloudScaling
from openpoints.dataset import get_features_by_keys


# ==========================================
# 1. 辅助工具函数
# ==========================================

def merge_list_of_dicts(data, dim=0):
    result = {key: torch.concat([d[key] for d in data], dim=dim) for key in data[0].keys()}
    return result


def split_to_batches(merged_dict, batch_size):
    total_samples = next(iter(merged_dict.values())).shape[0]
    num_batches = math.ceil(total_samples / batch_size)
    result = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_dict = {key: tensor[start_idx:end_idx] for key, tensor in merged_dict.items()}
        result.append(batch_dict)
    return result


def match_labels(pred, target):
    """ 匈牙利算法对齐传统聚类标签 """
    pred = pred.copy()
    c_labels, t_labels = np.unique(pred), np.unique(target)
    cost = np.zeros((len(c_labels), len(t_labels)), dtype=int)
    for i, c in enumerate(c_labels):
        for j, t in enumerate(t_labels):
            cost[i, j] = -np.sum((pred == c) & (target == t))
    ri, ci = linear_sum_assignment(cost)
    mapping = {c_labels[r]: t_labels[c] for r, c in zip(ri, ci)}
    return np.array([mapping.get(x, x) for x in pred], dtype=np.int32)


# ==========================================
# 2. 算法解耦实现
# ==========================================

def run_deep_learning_batch_inference(pts_orig, normal_orig, cls_idx, cfg_path, ckpt_path):
    """
    深度学习推理：包含打乱、补全、分批推理、截断及顺序恢复
    """
    num_points = 2048
    len_data = len(pts_orig)

    # --- A. 打乱与补全逻辑 ---
    idx = np.random.permutation(np.arange(len_data))
    # 补全到 num_points 的整数倍
    pad_size = num_points - len_data % num_points if len_data % num_points != 0 else 0
    # 按照你的逻辑：idx = np.concatenate([idx, idx[:num_points - len(idx) + (len(idx) // num_points) * num_points]])
    idx_padded = np.concatenate([idx, idx[:pad_size]])

    pts = pts_orig[idx_padded]
    normal = normal_orig[idx_padded]

    # --- B. 模型初始化 ---
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    model = build_model_from_cfg(cfg.model).cuda()
    model.load_state_dict(torch.load(ckpt_path)['model'])
    model.eval()

    data_transform1 = PointCloudCenterAndNormalize()
    data_transform2 = PointCloudScaling([0.8, 1.2])

    # 准备数据块
    datas = []
    for i in range(0, len(pts), num_points):
        p_chunk = torch.as_tensor(pts[i:i + num_points], dtype=torch.float32, device='cuda')
        n_chunk = torch.as_tensor(normal[i:i + num_points], dtype=torch.float32, device='cuda')
        c_chunk = torch.as_tensor([cls_idx], dtype=torch.int64, device='cuda')

        data_ = {'pos': p_chunk, 'x': n_chunk, 'cls': c_chunk}
        data_ = data_transform1(data_)

        for key in data_.keys():
            data_[key] = data_[key].unsqueeze(0)
            # if key == 'pos': data_[key] = data_transform2(data_[key])

        data_['x'] = get_features_by_keys(data_, 'pos,x,heights')
        datas.append(data_)

    # --- C. 批处理推理 ---
    batches = split_to_batches(merge_list_of_dicts(datas), 32)
    all_preds = []
    with torch.no_grad():
        for batch_data in batches:
            logits = model(batch_data)
            all_preds.append(torch.argmax(logits, dim=1).flatten())

    # 拼接并转回 numpy
    clusters_padded = torch.concat(all_preds).cpu().numpy()

    # --- D. 截取与恢复顺序 ---
    # 1. 截取掉补全的部分
    clusters_shuffled = clusters_padded[:len_data]
    # 2. 恢复打乱前的顺序：利用 np.argsort 还原 idx
    restore_idx = np.argsort(idx)
    final_clusters = clusters_shuffled[restore_idx]

    del model
    torch.cuda.empty_cache()
    return final_clusters


def run_region_growing(pts_raw):
    dll = ctypes.cdll.LoadLibrary(
        r"D:\Program Files\Microsoft Visual Studio\Projects\ConsoleApplication3\x64\Release\ConsoleApplication3.dll")
    pts_c = np.ascontiguousarray(pts_raw, dtype=np.float64)
    clusters = np.zeros(len(pts_c), dtype=np.int32)
    dll.RegionGrowing.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    dll.RegionGrowing(pts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(pts_c),
                      clusters.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    return clusters

def run_ransac(pts_raw):
    dll = ctypes.cdll.LoadLibrary(r"D:\Program Files\Microsoft Visual Studio\Projects\CCCore\x64\Release\CCCore.dll")
    pts_c = np.ascontiguousarray(pts_raw, dtype=np.float64)
    clusters = np.zeros(len(pts_c), dtype=np.int32)
    dll.RansacDetect.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    dll.RansacDetect(pts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(pts_c),
                     clusters.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    return clusters


# ==========================================
# 3. 执行与可视化
# ==========================================

def main():
    for root_path, j, k in os.walk(r'E:\data\bylw_indpart\raw_data'):
        for cls_idx, l in enumerate(k):
            test_file = os.path.join(root_path, l)
            part_start = [0, 11, 16, 22, 28, 43, 47, 61, 67, 73, 77, 91, 104, 109, 121, 137, 143, 159, 169, 175, 180, 193, 205, 219, 233, 245, 249, 254, 260, 266, 270, 283, 294, 304, 316, 328, 342, 346, 358, 371, 379, 393, 397, 411, 425, 437, 449, 461, 476, 488, 491, 495, 499, 505, 509, 522, 530, 534, 549, 557, 561, 573, 585, 593, 599, 604, 610, 614, 626, 630, 637, 654, 666, 678, 695, 711, 728]
            seg_num = [11, 5, 6, 6, 15, 4, 14, 6, 6, 4, 14, 13, 5, 12, 16, 6, 16, 10, 6, 5, 13, 12, 14, 14, 12, 4, 5, 6,
                       6, 4, 13, 11, 10, 12, 12, 14, 4, 12, 13, 8, 14, 4, 14, 14, 12, 12, 12, 15, 12, 3, 4, 4, 6, 4, 13,
                       8, 4, 15, 8, 4, 12, 12, 8, 6, 5, 6, 4, 12, 4, 7, 17, 12, 12, 17, 16, 17, 6]

            data = np.loadtxt(test_file).astype(np.float32)
            pts = data[:, :3]
            normal = data[:, 3:6]
            label_gt = data[:, 6].astype(np.int32)

            # 推理
            print("Computing RG...")
            label_rg = match_labels(run_region_growing(pts.copy()), label_gt)

            print("Computing RANSAC...")
            label_rs = match_labels(run_ransac(pts.copy()), label_gt)

            print("Inference Baseline...")
            cfg_b = r'E:\毕业论文\DGGAA2\log\indpart\indpart-train-baseline-ngpus1-seed8796-20260127-145338-XnByMnq9qqihnZj89MHRG3\baseline.yaml'
            ckpt_b = r'E:\毕业论文\DGGAA2\log\indpart\indpart-train-baseline-ngpus1-seed8796-20260127-145338-XnByMnq9qqihnZj89MHRG3\checkpoint\indpart-train-baseline-ngpus1-seed8796-20260127-145338-XnByMnq9qqihnZj89MHRG3_ckpt_best.pth'
            label_b = run_deep_learning_batch_inference(pts.copy(), normal.copy(), cls_idx, cfg_b, ckpt_b) - part_start[cls_idx]

            print("Inference Ours...")
            cfg_o = r'E:\毕业论文\DGGAA2\log\indpart\indpart-train-dggaa-ngpus1-seed6640-20260126-232338-GaX726k2gBmUgpbxHzcG4f\dggaa.yaml'
            ckpt_o = r'E:\毕业论文\DGGAA2\log\indpart\indpart-train-dggaa-ngpus1-seed6640-20260126-232338-GaX726k2gBmUgpbxHzcG4f\checkpoint\indpart-train-dggaa-ngpus1-seed6640-20260126-232338-GaX726k2gBmUgpbxHzcG4f_ckpt_best.pth'
            label_o =run_deep_learning_batch_inference(pts.copy(), normal.copy(), cls_idx, cfg_o, ckpt_o) - part_start[cls_idx]

            label_rg[label_rg <0] = seg_num[cls_idx]-1
            label_rs[label_rs <0] = seg_num[cls_idx]-1
            label_b[label_b<0]=seg_num[cls_idx]-1
            label_o[label_o<0]=seg_num[cls_idx]-1
            label_rg[label_rg >= seg_num[cls_idx]] = seg_num[cls_idx]-1
            label_rs[label_rs >= seg_num[cls_idx]] = seg_num[cls_idx]-1
            label_b[label_b>=seg_num[cls_idx]]=seg_num[cls_idx]-1
            label_o[label_o>=seg_num[cls_idx]]=seg_num[cls_idx]-1

            results_labels = [label_rg, label_rs, label_b, label_o]
            model_names = ["RegionGrowing", "RANSAC", "Baseline", "Ours"]

            print("\n" + "=" * 30)
            print(f"Test File: {os.path.basename(test_file)}")
            print("-" * 30)

            for name, pred in zip(model_names, results_labels):
                # 计算预测正确的点数比例
                accuracy = np.mean(pred == label_gt) * 100
                print(f"{name:<15} Accuracy: {accuracy:>6.2f}%")
            print("=" * 30 + "\n")

            # 可视化展示
            plotter = pv.Plotter(shape=(1, 5), window_size=(1900, 450))
            plotter.set_background("white")
            res_list = [label_gt, label_rg, label_rs, label_b, label_o]
            titles = ["GT", "RegionGrowing", "RANSAC", "Baseline", "Ours"]

            # 对显示点云做归一化
            pts_vis = (pts - np.mean(pts, axis=0)) / np.max([-np.min(pts), np.max(pts)])

            for i in range(5):
                plotter.subplot(0, i)
                cloud = pv.PolyData(pts_vis)
                print(set(res_list[i]))
                cloud["labels"] = res_list[i]
                plotter.add_text(titles[i], font_size=10, color="black")
                plotter.add_mesh(cloud, scalars="labels", cmap="tab20", categories=False,
                                 point_size=8, render_points_as_spheres=True, show_scalar_bar=False,clim=[0,seg_num[cls_idx]])

            plotter.link_views()
            plotter.show()


if __name__ == "__main__":
    main()
