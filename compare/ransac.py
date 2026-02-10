# coding=utf-8
import ctypes
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import random
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm


def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed()
times = 0
save_target = []
save_predict = []


def match_clusters_to_target(clusters, target):
    """
    将clusters的标签映射到target标签，使得尽量对齐。

    clusters: ndarray, shape (n,), 聚类结果
    target: ndarray, shape (n,), 真实标签

    返回映射后的clusters
    """
    clusters = clusters.copy()
    cluster_labels = np.unique(clusters)
    target_labels = np.unique(target)

    # 构建代价矩阵: cost[i,j] = 负的正确匹配数量（因为linear_sum_assignment是最小化）
    cost_matrix = np.zeros((len(cluster_labels), len(target_labels)), dtype=int)
    for i, c in enumerate(cluster_labels):
        for j, t in enumerate(target_labels):
            cost_matrix[i, j] = -np.sum((clusters == c) & (target == t))

    # Hungarian算法求最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 构建映射
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[cluster_labels[r]] = target_labels[c]

    # 对没有匹配的cluster直接保留原标签
    new_clusters = np.array([mapping.get(c, c) for c in clusters], dtype=clusters.dtype)
    return new_clusters


# 加载 DLL
dll = ctypes.cdll.LoadLibrary(r"D:\Program Files\Microsoft Visual Studio\Projects\CCCore\x64\Release\CCCore.dll")

# 函数类型声明
dll.RansacDetect.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # xyz
    ctypes.c_size_t,  # npoints
    ctypes.POINTER(ctypes.c_int)  # clusters
]
dll.RansacDetect.restype = ctypes.c_int

acc = []
test_list = []
for i, j, k in os.walk(r'E:\data\bylw_indpart\raw_data'):
    for l in k:
        test_list.append(os.path.join(i, l))
print(test_list)
for test_name in tqdm(test_list):
    data = np.loadtxt(test_name).astype(np.float32)
    pts = data[:, :3].astype(np.float64)
    target = data[:, 6].astype(np.int64)
    clusters = np.zeros(pts.shape[0], dtype=np.int32)
    s = time.time()
    ret = dll.RansacDetect(
        pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pts.shape[0],
        clusters.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    times += time.time() - s
    clusters = match_clusters_to_target(clusters, target)
    save_target.append(target.astype(np.int16))
    save_predict.append(clusters.astype(np.int16))
    acc.append(np.sum(clusters == target) / len(target))

print('times:', times/77)
result = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'jaccard': []}
y_trues = save_target
y_preds = save_predict
for i in tqdm(range(len(y_trues))):
    y_true = y_trues[i]
    y_pred = y_preds[i]
    result['accuracy'].append(accuracy_score(y_true, y_pred))
    result['precision'].append(precision_score(y_true, y_pred, average='macro', zero_division=1))
    result['recall'].append(recall_score(y_true, y_pred, average='macro', zero_division=1))
    result['f1'].append(f1_score(y_true, y_pred, average='macro', zero_division=1))
    result['jaccard'].append(jaccard_score(y_true, y_pred, average='macro', zero_division=1))

for key in result.keys():
    result[key] = np.mean(result[key])
print(result)

# times: 0.8139427172673213
# {'accuracy': 0.9138867087237381, 'precision': 0.7655161015869104, 'recall': 0.8526865601680371, 'f1': 0.7134948610273527, 'jaccard': 0.6593930919096614}
