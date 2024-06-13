import os
import json
import math
import imageio 
import random
import numpy as np

import cv2

import torch
from torch.utils.data import Dataset

import pdb

class NeRFSynthesicDataset(Dataset):
    def __init__(self, data_path, cfg):
        super(NeRFSynthesicDataset, self).__init__()
        """
        Args:
            data_pata (str): data directory.
            split (str): train, val, test
        """
        # dataset configs
        self.data_path = data_path
        self.near = cfg['near']
        self.far = cfg['far']
        self.half_size = cfg['half_size']
        
        # load transforms
        frames = []
        split = ['train', 'val', 'test']
        for s in split:
            with open(os.path.join(data_path, 'transforms_{}.json'.format(s)), 'r') as file:
                transfroms = json.load(file)
            camera_angle_x = transfroms['camera_angle_x']
            frames += transfroms['frames']
        random.shuffle(frames)
        self.frame_set = frames
        # load an image to get H/W
        file_path = os.path.join(self.data_path, frames[0]['file_path'][2:]) + '.png'
        image = imageio.v2.imread(file_path)

        # camera intrinsics
        self.H = image.shape[0]
        self.W = image.shape[1]
        self.focal = 0.5 * self.W / math.tan(0.5 * camera_angle_x)
        if self.half_size:
            self.H =  self.H // 2
            self.W =  self.W // 2
            self.focal =  self.focal / 2.0

    def create_bounding_box(self):
        def create_boundary(c2w):
            boundary = []
            bd_point = [(0, 0), (0, self.H - 1), (self.W - 1, 0), (self.W - 1, self.H - 1)]
            for i, point in enumerate(bd_point):
                ray = torch.tensor(
                    [(point[0] - self.W * 0.5) / self.focal, -(point[1] - self.H * 0.5) / self.focal, -1.0], 
                    dtype=torch.float32
                )
                ray = ray @ c2w[:3, :3].T
                # ray = ray / torch.norm(ray, dim=-1, keepdim=True)
                pts = c2w[:3, -1]
                boundary.append(pts + ray * self.near)
                boundary.append(pts + ray * self.far)
            return boundary
        
        bb_min = torch.tensor([torch.inf, torch.inf, torch.inf], dtype=torch.float32)
        bb_max = torch.tensor([-torch.inf, -torch.inf, -torch.inf], dtype=torch.float32)
    
        for idx in range(len(self.frame_set)):
            c2w = torch.tensor(self.frame_set[idx]['transform_matrix'], dtype=torch.float32)
            boundary = create_boundary(c2w)
            for i in range(len(boundary)):
                for k in range(3):
                    if bb_max[k] < boundary[i][k]:
                        bb_max[k] = boundary[i][k]
                    if bb_min[k] > boundary[i][k]:
                        bb_min[k] = boundary[i][k]
        # expand
        expand = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        bb_max += expand
        bb_min -= expand

        return torch.cat((bb_min, bb_max), dim=-1)
        
    def __len__(self):
        return len(self.frame_set)
    
    def __getitem__(self, index):
        # load ground truth image
        file_path = os.path.join(self.data_path, self.frame_set[index]['file_path'][2:]) + '.png'
        image = imageio.v2.imread(file_path)
        
        # half-size
        if self.half_size:
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
        image = torch.tensor(image, dtype=torch.float32)
        
        # generate rays
        c2w = torch.tensor(self.frame_set[index]['transform_matrix'], dtype=torch.float32)
        x, y = torch.meshgrid(
            torch.arange(self.W, dtype=torch.float32), 
            torch.arange(self.H, dtype=torch.float32), 
            indexing='xy'
        )
        dirs = torch.stack([(x - self.W * 0.5) / self.focal, -(y - self.H * 0.5) / self.focal, -torch.ones_like(x)], -1)
        dirs_world = dirs @ c2w[:3, :3].T
        # dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
        #! normalize了，不然有点怪。尝试了HashNeRF发现norm不norm差别不大
        pts =  c2w[:3, -1].expand(self.H, self.W, 3)
        rays = torch.cat([dirs_world, pts], dim=-1)
        # both tensors but not on cuda
        return {'rays' : rays, 'gt_image' : image}
    
#! TODO
class ColmapDataset(Dataset):
    def __init__(self, data_path, cfg) -> None:
        super(ColmapDataset, self).__init__()
        """
        Args:
            data_pata (str): data directory.
            split (str): train, val, test
        """
        
    def __len__(self):
        pass

    def __getitem__(self, index):
        return None

def make_dataset(cfg, data_path):
    if cfg['type'] == 'nerf_synthesic':
        return NeRFSynthesicDataset(data_path, cfg)
    elif cfg['type'] == 'colmap':
        return ColmapDataset(data_path, cfg)
    else :
        raise NotImplementedError("[ERROR] No such data type: {}.".format(data_path))
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x
