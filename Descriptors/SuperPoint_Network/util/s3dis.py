import os
import numpy as np
import time
import math

from torch.utils.data import Dataset

class S3DIS(Dataset):
    def __init__(self, split='train', data_dict=None, transform=None):
        super().__init__()
        if split not in ['train']:
            print("train.py S3DIS only support 'train'!")
            exit()
        self.for_debug = data_dict["for_debug"]
        self.data_root = data_dict["data_root"]
        self.num_point = data_dict["num_point"]
        self.block_size = data_dict["block_size"]
        self.sample_rate = data_dict["sample_rate"]
        self.test_area = data_dict["test_area"]
        self.minpoints = data_dict["minpoints"]
        #self.voxel_resolution = data_dict["voxel_resolution"]
        self.transform = transform
   
        rooms = sorted(os.listdir(self.data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(self.test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(self.test_area) in room]
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        #se = set()
        start_time = time.time()
        #zmax = -1.0
        #zflag = False
        for room_name in rooms_split:
            room_path = os.path.join(self.data_root, room_name)
            #print('room_path: {}'.format(room_path))
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            #tz = np.max(points[:, 2])
            #if tz > zmax:
            #    zmax = tz
            #if np.min(points[:, 2]) < 0.0:
            #    zflag = True
            #for val in labels:
            #    se.add(val)
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            #print(coord_min, coord_max)
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
            if self.for_debug: break
        #print('se: ', len(se))
        #print('zmax {}'.format(zmax))
        #print('zflag {}'.format(zflag))
        print('read S3DIS data time: {:2f} min'.format((time.time()-start_time)/60.))
        print('total points: {}'.format(np.sum(num_point_all)))
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * self.sample_rate / self.num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
            if self.for_debug: break
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            #if point_idxs.size > 1024:
            if point_idxs.size > self.minpoints:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        #delta_z = math.ceiling(np.max(selected_points[:, 2]) - np.min(selected_points[:, 2]))
        #print(delta_z) 

        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


class S3DIS_V1(Dataset):
    def __init__(self, split='train', data_dict=None, transform=None):
        super().__init__()
        if split not in ['train']:
            print("S3DIS only support 'train'!")
            exit()
        self.for_debug = data_dict["for_debug"]
        self.data_root = data_dict["data_root"]
        self.num_point = data_dict["num_point"]
        self.block_size = data_dict["block_size"]
        self.sample_rate = data_dict["sample_rate"]
        self.test_area = data_dict["test_area"]
        self.minpoints = data_dict["minpoints"]
        self.rs = data_dict["rs"]
        
        self.scale = data_dict["scale"]
        self.aug_scale_anisotropic = data_dict["aug_scale_anisotropic"]
        self.aug_symmetries = data_dict["aug_symmetries"]
        self.aug_rotation = data_dict["aug_rotation"]
        self.aug_scale_min = data_dict["aug_scale_min"]
        self.aug_scale_max = data_dict["aug_scale_max"]
        self.aug_noise = data_dict["aug_noise"]
        self.aug_color = data_dict["aug_color"]
        
        self.transform = transform
        rooms = sorted(os.listdir(self.data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(self.test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(self.test_area) in room]
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        for room_name in rooms_split: 
            room_path = os.path.join(self.data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
            if self.for_debug: break
        print(np.sum(num_point_all))
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * self.sample_rate / self.num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
            if self.for_debug: break
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            #if point_idxs.size > 1024:
            if point_idxs.size > self.minpoints:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        
        if self.aug_rotation:
            # choose a random angle for points
            theta = np.random.uniform(0.0, 2*np.pi, 1)
            # rotation matrices
            c, s = np.cos(theta), np.sin(theta)
            cs0 = 0.0
            cs1 = 1.0
            #R = np.array([c, -s, cs0, s, c, cs0, cs0, cs0, cs1])
            R = np.array([[c, -s, cs0], 
                          [s, c, cs0],
                          [cs0, cs0, cs1]], dtype=np.float64)
            #print(R.shape, R.dtype)
            #tmp = current_points[:, 0:3]
            #print(tmp.shape, tmp.dtype)
            #tmp = np.matmul(tmp, R)   # Nx3 X 3x3 -> Nx3
            current_points[:, 0:3] = np.matmul(current_points[:, 0:3], R)  #Nx3
            #exit()
        
        if self.scale:
            if self.aug_scale_min is not None:
                min_s = self.aug_scale_min
            if self.aug_scale_max is not None:
                max_s = self.aug_scale_max

            if self.aug_scale_anisotropic is not None:
                s = np.random.uniform(min_s, max_s, 3)
            else:
                s = np.random.uniform(min_s, max_s, 1)
            symmetrics = []
            for i in range(3):
                if self.aug_symmetries[i]:
                    symmetrics.append(np.round(np.random.uniform(0.0, 1.0, 1)) * 2 - 1)
                else:
                    symmetrics.append(1.0)
            symmetrics = np.array(symmetrics, dtype=np.float64)
            #print(s, s.shape, s.dtype)
            #print(symmetrics, symmetrics.shape, symmetrics.dtype)
            #exit()
            s = s * symmetrics
            #print(s, s.shape, s.dtype)
            #exit()
            current_points[:, 0] = current_points[:, 0] * s[0]
            current_points[:, 1] = current_points[:, 1] * s[0]
            current_points[:, 2] = current_points[:, 2] * s[0]
        
        if self.aug_noise is not None:
            noise = np.random.normal(0.0, self.aug_noise, self.num_point*3)
            noise = np.reshape(noise, (self.num_point, 3))
            #print('noise', noise.shape, noise.dtype)
            #print(noise[0:5, :])
            #exit()
            current_points[:, 0:3] = current_points[:, 0:3] + noise
            
        if self.aug_color is not None:
            col = np.random.uniform(0.0, 1.0, self.num_point*3)
            col[col < self.aug_color] = self.aug_color
            col = np.reshape(col, (self.num_point, 3))
            #print('col', col.shape, col.dtype)
            #print(col[:5, :])
            #exit()
            current_points[:, 3:6] = current_points[:, 3:6] * col


        current_labels = labels[selected_point_idxs]


        if self.rs: # 4096: 1024, 256, 64, 16
            l1 = np.arange(4096)
            np.random.shuffle(l1)
            l1 = l1[:1024]

            l2 = np.arange(1024)
            np.random.shuffle(l2)
            l2 = l2[:256]

            l3 = np.arange(256)
            np.random.shuffle(l3)
            l3 = l3[:64]

            l4 = np.arange(64)
            np.random.shuffle(l4)
            l4 = l4[:16]
        else:
            l1 = l2 = l3 = l4 = np.array([0])

        if self.transform is not None:
            current_points, current_labels, l1, l2, l3, l4 = self.transform(current_points, current_labels, l1, l2, l3, l4) 
        
        return current_points, current_labels, l1, l2, l3, l4

    def __len__(self):
        return len(self.room_idxs)

##############################################################################

class S3DIS_pointweb(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None, minpoints=1024):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.minpoints = minpoints
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        #se = set()
        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            #print('room_path: {}'.format(room_path))
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            #for val in labels:
            #    se.add(val)
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            #print(coord_min, coord_max)
            #print(type(points[0, 0]))
            #print(type(points[0, 3]))
            #print(type(labels[0]))
            #for i in range(points.shape[0]):
            #    if points[i, 3] == 141 and points[i, 4] == 147 and points[i, 5] == 135:
            #        print(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5], labels[i])
            #for val in points:
            #    print(val)
            #exit()
            #print(coord_min)
            #print(coord_max)
            #exit()
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
            #break
        #print('se: ', len(se))
        print(np.sum(num_point_all))
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
            #break
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            #if point_idxs.size > 1024:
            if point_idxs.size > self.minpoints:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    import yaml
    #DATA = yaml.safe_load(open('../config/s3dis_pointnet2_rs_20191230_a.yaml', 'r'))
    DATA = yaml.safe_load(open('../config/s3dis_pointnet2_20200425_a.yaml', 'r'))
    DATA["for_debug"] = True
    #DATA["rs"] = True
    print(DATA["train_gpu"])
    # exit()
    num_workers = 8
    
    import transform
    # train_transform = transform.Compose_withIDX([transform.ToTensor_withIDX()])
    train_transform = None
    point_data = S3DIS_V1(split='train', data_dict=DATA, transform=train_transform)
    a, b, l1, l2, l3, l4 = point_data.__getitem__(0)
    if DATA["rs"]:
        print(a.shape, b.shape, l1.shape, l2.shape, l3.shape, l4.shape)
    else:
        print(a.shape, b.shape)


    """
    data_root = '/test/dataset/3d_datasets/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 1.0

    point_data = S3DIS(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    exit()
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    print(len(train_loader))
    exit()
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()
    """
