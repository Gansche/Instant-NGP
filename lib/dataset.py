import os
import json
import math
import imageio 
import random

import torch
from torch.utils.data import Dataset

import pdb

class NeRFSynthesicDataset(Dataset):
    def __init__(self, data_path):
        super(NeRFSynthesicDataset, self).__init__()
        """
        Args:
            data_pata (str): data directory.
            split (str): train, val, test
        """
        # dataset configs
        self.data_path = data_path
        
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
        
    def __len__(self):
        return len(self.frame_set)
    
    def __getitem__(self, index):
        # load ground truth image
        file_path = os.path.join(self.data_path, self.frame_set[index]['file_path'][2:]) + '.png'
        image = imageio.v2.imread(file_path)
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
        dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
        pts =  c2w[:3, -1].expand(self.H, self.W, 3)
        rays = torch.cat([dirs_world, pts], dim=-1)
        
        # both tensors but not on cuda
        item = {
            'rays' : rays,  # (W, H, d+o=6)
            'gt_image' : image
        }
        return item
    
#! TODO
class ColmapDataset(Dataset):
    def __init__(self, data_path, split='train') -> None:
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
        return NeRFSynthesicDataset(data_path)
    elif cfg['type'] == 'colmap':
        return ColmapDataset(data_path)
    else :
        raise NotImplementedError("[ERROR] No such data type: {}.".format(data_path))
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x
