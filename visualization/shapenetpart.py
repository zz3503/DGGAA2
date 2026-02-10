# coding=utf-8
import os
import numpy as np
import torch
import open3d as o3d
from openpoints.models import build_model_from_cfg
from openpoints.utils import cal_model_parm_nums, EasyConfig
from openpoints.models.layers.subsample import furthest_point_sample
from openpoints.transforms.point_transform_cpu import PointsToTensor
from openpoints.transforms.point_transformer_gpu import PointCloudCenterAndNormalize, PointCloudScaling
from openpoints.dataset import get_features_by_keys
import time
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm
import math
import pyvista as pv


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

data_transform1 = PointCloudCenterAndNormalize(gravity_dim=1)

cfg = EasyConfig()
cfg.load(
    r'E:\毕业论文\DGGAA2\log\shapenetpart\shapenetpart-train-pointnext-s_c64-ngpus4-seed3050-20260108-010351-GG4M5SotyLeHoif2DMoanb20260108-132740-GyZhm3hURMd8fAXTT7NxfK\dggaa.yaml',
    recursive=True)
model = build_model_from_cfg(cfg.model).cuda()
model_size = cal_model_parm_nums(model)
print('Number of params: %.4f M' % (model_size / 1e6))

checkpoint = torch.load(
    r'E:\毕业论文\DGGAA2\log\shapenetpart\shapenetpart-train-pointnext-s_c64-ngpus4-seed3050-20260108-010351-GG4M5SotyLeHoif2DMoanb20260108-132740-GyZhm3hURMd8fAXTT7NxfK\checkpoint\shapenetpart-train-pointnext-s_c64-ngpus4-seed3050-20260108-010351-GG4M5SotyLeHoif2DMoanb_ckpt_best.pth', )

ckpt_state = checkpoint['model']
model.load_state_dict(ckpt_state)
model.eval()
acc = []

test_list = []
root_path = r'E:\data\ShapeNetPart\shapenetcore_partanno_segmentation_benchmark_v0_normal'
test_list.append(os.path.join(root_path, '02691156/a1708ad923f3b51abbf3143b1cb6076a.txt'))
test_list.append(os.path.join(root_path, '02773838/4e4fcfffec161ecaed13f430b2941481.txt'))
test_list.append(os.path.join(root_path, '02954340/c7122c44495a5ac6aceb0fa31f18f016.txt'))
test_list.append(os.path.join(root_path, '02958343/cb19594e73992a3d51008e496c6cfd2e.txt'))
test_list.append(os.path.join(root_path, '03001627/355fa0f35b61fdd7aa74a6b5ee13e775.txt'))
test_list.append(os.path.join(root_path, '03261776/e33d6e8e39a75268957b6a4f3924d982.txt'))
test_list.append(os.path.join(root_path, '03467517/d546e034a6c659a425cd348738a8052a.txt'))
test_list.append(os.path.join(root_path, '03624134/9d424831d05d363d870906b5178d97bd.txt'))
test_list.append(os.path.join(root_path, '03636649/b8c87ad9d4930983a8d82fc8a3e54728.txt'))
test_list.append(os.path.join(root_path, '03642806/4d3dde22f529195bc887d5d9a11f3155.txt'))
test_list.append(os.path.join(root_path, '03790512/9d3b07f4475d501e8249f134aca4c817.txt'))
test_list.append(os.path.join(root_path, '03797390/10f6e09036350e92b3f21f1137c3c347.txt'))
test_list.append(os.path.join(root_path, '03948459/b1bbe535a833635d91f9af3df5b0c8fc.txt'))
test_list.append(os.path.join(root_path, '04099429/15474cf9caa757a528eba1f0b7744e9.txt'))
test_list.append(os.path.join(root_path, '04225987/f5d7698b5a57d61226e0640b67de606.txt'))
test_list.append(os.path.join(root_path, '04379243/408c3db9b4ee6be2e9f3e9c758fef992.txt'))
print(test_list)

part_start = [0, 11, 16, 22, 30, 45, 49, 63, 71, 77, 81, 95, 108, 113, 125, 141, 148, 164, 174, 180, 185, 200, 213, 227,
              241, 253, 257, 265, 273, 281, 285, 300, 311, 321, 333, 345, 359, 363, 375, 388, 396, 410, 414, 428, 442,
              454, 467, 479, 494, 509, 512, 521, 525, 531, 537, 550, 558, 562, 577, 585, 589, 601, 613, 623, 631, 636,
              644, 648, 661, 665, 672, 679, 691, 698, 715, 731, 748]
num_points = 2048
cls_parts = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
             'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37],
             'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
             'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40], 'chair': [12, 13, 14, 15], 'knife': [22, 23]}


def main():
    global times
    for cls, test_name in enumerate(test_list):
        data = np.loadtxt(test_name).astype(np.float32)[:2048, :]

        pts = data[:, :3].astype(np.float64)
        pts = (pts - np.mean(pts, axis=0)) / np.max([-np.min(pts), np.max(pts)])
        normal = data[:, 3:6].astype(np.float64)
        target = data[:, 6].astype(np.int64)
        cls = np.array([cls], dtype=np.int32)
        pts = torch.as_tensor(pts, dtype=torch.float32, device='cuda')
        normal = torch.as_tensor(normal, dtype=torch.float32, device='cuda')
        cls = torch.as_tensor(cls, dtype=torch.int64, device='cuda')
        clusters = []
        datas = []
        data_ = {'pos': pts, 'x': normal, 'cls': cls}
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

        pts = pts.cpu().numpy()

        valid_labels = np.array(sorted(list(cls_parts.values()))[cls], dtype=np.int64)
        extra_pts = np.tile(pts[0], (len(valid_labels), 1))
        pts = np.vstack([pts, extra_pts])
        target = np.concatenate([target, valid_labels])
        clusters = np.concatenate([clusters, valid_labels])

        plotter = pv.Plotter(shape=(1, 2))
        cloud1 = pv.PolyData(pts)
        cloud1["labels"] = target
        cloud2 = pv.PolyData(pts)
        cloud2["labels"] = clusters

        plotter.subplot(0, 0)
        plotter.add_text("Label 1 (Ground Truth)", font_size=10)
        plotter.add_mesh(
            cloud1,
            scalars="labels",
            cmap="tab10",
            categories=True,
            point_size=10.0,
            render_points_as_spheres=True,
            name="mesh1",
            clim=[np.min(clusters), np.max(clusters)]
        )
        plotter.subplot(0, 1)
        plotter.add_text("Label 2 (Prediction)", font_size=10)
        plotter.add_mesh(
            cloud2,
            scalars="labels",
            cmap="tab10",
            categories=True,
            point_size=10.0,
            render_points_as_spheres=True,
            name="mesh2",
            clim=[np.min(clusters), np.max(clusters)]
        )
        plotter.link_views()
        print(test_name.split('\\')[-1], np.sum(clusters == target) / len(target))
        plotter.show()


main()
