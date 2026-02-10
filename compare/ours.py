# coding=utf-8
import os
import numpy as np
import torch
import open3d as o3d
from openpoints.models import build_model_from_cfg
from openpoints.utils import cal_model_parm_nums, EasyConfig
from openpoints.models.layers.subsample import furthest_point_sample
from openpoints.transforms.point_transform_cpu import PointsToTensor
from openpoints.transforms.point_transformer_gpu import PointCloudCenterAndNormalize, PointCloudScaling, \
    PointCloudHeightsNormalize
from openpoints.dataset import get_features_by_keys
import time
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm
import math


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


def split_to_batches(merged_dict, batch_size):
    total_batch = next(iter(merged_dict.values())).shape[0]
    num_batches = math.ceil(total_batch / batch_size)
    result = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_batch)
        batch_dict = {
            key: tensor[start_idx:end_idx]
            for key, tensor in merged_dict.items()
        }
        result.append(batch_dict)
    return result


def merge_list_of_dicts(data, dim=0):
    result = {}
    for key in data[0].keys():
        tensors = [d[key] for d in data]
        result[key] = torch.concat(tensors, dim=dim)
    return result


set_random_seed()
times = 0
save_target = []
save_predict = []

data_transform1 = PointCloudCenterAndNormalize()


def random_rotate_single_axis(pts: np.ndarray,
                              normal: np.ndarray,
                              axis: int = 2,
                              angle: float = None):
    """
    仅绕指定轴随机旋转
    axis : 0=X, 1=Y, 2=Z
    angle: 给定角度（rad）；None 则随机 0~2π
    return -> pts_rot, normal_rot
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normal)

    if angle is None:
        angle = np.random.rand() * 2 * np.pi

    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:  # 绕 X
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
    elif axis == 1:  # 绕 Y
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
    else:  # 绕 Z
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])

    pcd.rotate(R, center=np.zeros(3))
    return np.asarray(pcd.points), np.asarray(pcd.normals)


cfg = EasyConfig()
cfg.load(
    r'E:\毕业论文\DGGAA2\log\indpart\indpart-train-dggaa-ngpus1-seed6640-20260126-232338-GaX726k2gBmUgpbxHzcG4f\dggaa.yaml',
    recursive=True)

model = build_model_from_cfg(cfg.model).cuda()
model_size = cal_model_parm_nums(model)
print('Number of params: %.4f M' % (model_size / 1e6))

checkpoint = torch.load(
    r'E:\毕业论文\DGGAA2\log\indpart\indpart-train-dggaa-ngpus1-seed6640-20260126-232338-GaX726k2gBmUgpbxHzcG4f\checkpoint\indpart-train-dggaa-ngpus1-seed6640-20260126-232338-GaX726k2gBmUgpbxHzcG4f_ckpt_best.pth', )

ckpt_state = checkpoint['model']
model.load_state_dict(ckpt_state)
model.eval()
acc = []
test_list = []
for i, j, k in os.walk(r'E:\data\bylw_indpart\raw_data'):
    for l in k:
        test_list.append(os.path.join(i, l))
print(test_list)
part_start = [0, 11, 16, 22, 28, 43, 47, 61, 67, 73, 77, 91, 104, 109, 121, 137, 143, 159, 169, 175, 180, 193, 205, 219,
              233, 245, 249, 254, 260, 266, 270, 283, 294, 304, 316, 328, 342, 346, 358, 371, 379, 393, 397, 411, 425,
              437, 449, 461, 476, 488, 491, 495, 499, 505, 509, 522, 530, 534, 549, 557, 561, 573, 585, 593, 599, 604,
              610, 614, 626, 630, 637, 654, 666, 678, 695, 711, 728]

num_points = 2048


def main():
    global times
    for cls, test_name in tqdm(enumerate(test_list), total=len(test_list)):
        data = np.loadtxt(test_name).astype(np.float32)
        s = time.time()

        len_data = len(data)
        pts = data[:, :3].astype(np.float64)
        pts = (pts - np.mean(pts, axis=0)) / np.max([-np.min(pts), np.max(pts)])
        normal = data[:, 3:6].astype(np.float64)
        # pts, normal = random_rotate_single_axis(pts, normal, 2)
        target = data[:, 6].astype(np.int64)
        # idx = furthest_point_sample(torch.as_tensor(pts).float().cuda().unsqueeze(0), len(pts))[0].cpu().numpy()
        idx = np.random.permutation(np.arange(len_data))
        idx = np.concatenate([idx, idx[:num_points - len(idx) + (len(idx) // num_points) * num_points]])
        pts = pts[idx]
        normal = normal[idx]
        target = target[idx]
        cls = np.array([cls], dtype=np.int32)
        pts = torch.as_tensor(pts, dtype=torch.float32, device='cuda')
        normal = torch.as_tensor(normal, dtype=torch.float32, device='cuda')
        cls = torch.as_tensor(cls, dtype=torch.int64, device='cuda')
        clusters = []
        datas = []
        for i in range(0, len(pts), num_points):
            data_ = {'pos': pts[i:i + num_points], 'x': normal[i:i + num_points], 'cls': cls}
            data_ = data_transform1(data_)
            for key in data_.keys():
                data_[key] = data_[key].unsqueeze(0)
            data_['x'] = get_features_by_keys(data_, 'pos,x,heights')
            datas.append(data_)
        datas = split_to_batches(merge_list_of_dicts(datas), 32)
        with torch.no_grad():
            for data in datas:
                logits = model(data)
                logits = torch.argmax(logits, dim=1).flatten()
                clusters.append(logits)
        clusters = torch.concat(clusters).cpu().numpy()
        clusters = clusters - part_start[cls.item()]

        times += time.time() - s
        save_target.append(target.astype(np.int16)[:len_data])
        save_predict.append(clusters.astype(np.int16)[:len_data])
        acc.append(np.sum(clusters == target) / len(target))


main()

print('times :', times / 77)
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
# times : 1.2753541995952655
# {'accuracy': 0.8936597980520125, 'precision': 0.8307798605042584, 'recall': 0.8890691813154992, 'f1': 0.7917245299778042, 'jaccard': 0.7513464407306353}
