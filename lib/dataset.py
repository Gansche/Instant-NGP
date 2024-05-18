import os
import json
import math
import imageio 
import random
import pdb

import torch
from torch.utils.data import Dataset

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
        self.H = image[0].shape[0]
        self.W = image[0].shape[1]
        self.focal = 0.5 * self.W / math.tan(0.5 * camera_angle_x)
        
    def __len__(self):
        return len(self.frame_set)
    
    def __getitem__(self, index):
        file_path = os.path.join(self.data_path, self.frame_set[index]['file_path'][2:]) + '.png'
        image = imageio.v2.imread(file_path)
        
        # generate rays
        x, y = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        pts = torch.tensor(self.frame_set[index]['transform_matrix'][:3, -1]).float().expand(self.H * self.W)
        #!TODO
        
        #! warning: return a tensor but not on cuda
    
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
