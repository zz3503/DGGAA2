import os
import glob
import h5py
import json
import pickle
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps


@DATASETS.register_module()
class IndPartNormal(Dataset):
    classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24',
               '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40',
               '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56',
               '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '7', '8', '9', 'luowen1',
               'luowen2', 'mocakuai1', 'mocakuai2', 'mocakuai3', 'mocakuai4', 'wolunpan1', 'wolunpan2',
               'zhouchengxiutao']
    seg_num = [11, 5, 6, 6, 15, 4, 14, 6, 6, 4, 14, 13, 5, 12, 16, 6, 16, 10, 6, 5, 13, 12, 14, 14, 12, 4, 5, 6, 6, 4, 13, 11, 10, 12, 12, 14, 4, 12, 13, 8, 14, 4, 14, 14, 12, 12, 12, 15, 12, 3, 4, 4, 6, 4, 13, 8, 4, 15, 8, 4, 12, 12, 8, 6, 5, 6, 4, 12, 4, 7, 17, 12, 12, 17, 16, 17, 6]

    cls_parts = {
        '0': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        '1': [11, 12, 13, 14, 15],
        '10': [16, 17, 18, 19, 20, 21],
        '11': [22, 23, 24, 25, 26, 27],
        '12': [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        '13': [43, 44, 45, 46],
        '14': [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        '15': [61, 62, 63, 64, 65, 66],
        '16': [67, 68, 69, 70, 71, 72],
        '17': [73, 74, 75, 76],
        '18': [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
        '19': [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103],
        '2': [104, 105, 106, 107, 108],
        '20': [109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
        '21': [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136],
        '22': [137, 138, 139, 140, 141, 142],
        '23': [143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158],
        '24': [159, 160, 161, 162, 163, 164, 165, 166, 167, 168],
        '25': [169, 170, 171, 172, 173, 174],
        '26': [175, 176, 177, 178, 179],
        '27': [180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192],
        '28': [193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204],
        '29': [205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218],
        '3': [219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232],
        '30': [233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244],
        '31': [245, 246, 247, 248],
        '32': [249, 250, 251, 252, 253],
        '33': [254, 255, 256, 257, 258, 259],
        '34': [260, 261, 262, 263, 264, 265],
        '35': [266, 267, 268, 269],
        '36': [270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282],
        '37': [283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293],
        '38': [294, 295, 296, 297, 298, 299, 300, 301, 302, 303],
        '39': [304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315],
        '4': [316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327],
        '40': [328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341],
        '41': [342, 343, 344, 345],
        '42': [346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357],
        '43': [358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370],
        '44': [371, 372, 373, 374, 375, 376, 377, 378],
        '45': [379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392],
        '46': [393, 394, 395, 396],
        '47': [397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410],
        '48': [411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424],
        '49': [425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436],
        '5': [437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448],
        '50': [449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460],
        '51': [461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475],
        '52': [476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487],
        '53': [488, 489, 490],
        '54': [491, 492, 493, 494],
        '55': [495, 496, 497, 498],
        '56': [499, 500, 501, 502, 503, 504],
        '57': [505, 506, 507, 508],
        '58': [509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521],
        '59': [522, 523, 524, 525, 526, 527, 528, 529],
        '6': [530, 531, 532, 533],
        '60': [534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548],
        '61': [549, 550, 551, 552, 553, 554, 555, 556],
        '62': [557, 558, 559, 560],
        '63': [561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572],
        '64': [573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584],
        '65': [585, 586, 587, 588, 589, 590, 591, 592],
        '66': [593, 594, 595, 596, 597, 598],
        '67': [599, 600, 601, 602, 603],
        '7': [604, 605, 606, 607, 608, 609],
        '8': [610, 611, 612, 613],
        '9': [614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625],
        'luowen1': [626, 627, 628, 629],
        'luowen2': [630, 631, 632, 633, 634, 635, 636],
        'mocakuai1': [637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653],
        'mocakuai2': [654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665],
        'mocakuai3': [666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677],
        'mocakuai4': [678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694],
        'wolunpan1': [695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710],
        'wolunpan2': [711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727],
        'zhouchengxiutao': [728, 729, 730, 731, 732, 733]
    }
    cls2parts = []
    cls2partembed = torch.zeros(77, 734)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat

    def __init__(self,
                 data_root=r'E:\data\bylw_indpart\fps_data',
                 num_points=2048,
                 split='train',
                 class_choice=None,
                 use_normal=True,
                 shape_classes=77,
                 presample=False,
                 sampler='fps',
                 transform=None,
                 multihead=False,
                 **kwargs
                 ):
        self.npoints = num_points
        self.root = data_root
        self.transform = transform
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.use_normal = use_normal
        self.presample = presample
        self.sampler = sampler
        self.split = split
        self.multihead = multihead
        self.part_start = [0, 11, 16, 22, 28, 43, 47, 61, 67, 73, 77, 91, 104, 109, 121, 137, 143, 159, 169, 175, 180, 193, 205, 219, 233, 245, 249, 254, 260, 266, 270, 283, 294, 304, 316, 328, 342, 346, 358, 371, 379, 393, 397, 411, 425, 437, 449, 461, 476, 488, 491, 495, 499, 505, 509, 522, 530, 534, 549, 557, 561, 573, 585, 593, 599, 604, 610, 614, 626, 630, 637, 654, 666, 678, 695, 711, 728]

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if (
                        (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        if transform is None:
            self.eye = np.eye(shape_classes)
        else:
            self.eye = torch.eye(shape_classes)

        # in the testing, using the uniform sampled 2048 points as input
        # presample
        filename = os.path.join(data_root, 'processed',
                                f'{split}_{num_points}_fps.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data, self.cls = [], []
            npoints = []
            for cat, filepath in tqdm(self.datapath, desc=f'Sample IndPart {split} split'):
                cls = self.classes[cat]
                cls = np.array([cls]).astype(np.int64)
                data = np.loadtxt(filepath).astype(np.float32)
                npoints.append(len(data))
                data = torch.from_numpy(data).to(
                    torch.float32).cuda().unsqueeze(0)
                data = fps(data, num_points).cpu().numpy()[0]
                self.data.append(data)
                self.cls.append(cls)
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(os.path.join(data_root, 'processed'), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump((self.data, self.cls), f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data, self.cls = pickle.load(f)
                print(f"{filename} load successfully")

    def __getitem__(self, index):
        if not self.presample:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int64)
        else:
            data, cls = self.data[index], self.cls[index]
            point_set, seg = data[:, :6], data[:, 6].astype(np.int64)

        if 'train' in self.split:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[choice]
            seg = seg[choice]
        else:
            point_set = point_set[:self.npoints]
            seg = seg[:self.npoints]
        if not self.multihead:
            seg = seg + self.part_start[cls[0]]

        data = {'pos': point_set[:, 0:3],
                'x': point_set[:, 3:6],
                'cls': cls,
                'y': seg}

        """debug 
        from openpoints.dataset import vis_multi_points
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
        """
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    train = IndPartNormal(num_points=2048, split='trainval')
    test = IndPartNormal(num_points=2048, split='test')
    for dict in test:
        for i in dict:
            print(i, dict[i].shape)
        break
