import os
import json
import imageio 

from torch.utils.data import Dataset

class NeRFSynthesicDataset(Dataset):
    def __init__(self, data_path, split='train'):
        super(NeRFSynthesicDataset, self).__init__()
        """
        Args:
            data_pata (str): data directory.
            split (str): train, val, test
        """
        # dataset configs
        assert split in ['train', 'val', 'test']
        self.split = split
        self.data_path = data_path
        
        # load transforms
        with open(os.path.join(data_path, 'transforms_{}.json'.format(split)), 'r') as file:
            transfroms = json.load(file)
        self.camera_angle_x = transfroms['camera_angle_x']
        self.frames = transfroms['frames']
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        file_path = os.path.join(self.data_path, self.frames[index]['file_path'][2:]) + '.png'
        try:
            image = imageio.v2.imread(file_path)
        except:
            raise IOError('[ERROR] Image loading failed: {:s}'.format(file_path))
        
        #! TODO : image need transformation?
        #! TODO : on cuda?
        lib = {
            'camera_angle_x': self.camera_angle_x,
            'rotation': self.frames[index]['rotation'],
            'transform_matrix': self.frames[index]['transform_matrix'],
            'image': image
        }
        return lib
    
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

def make_dataset(cfg, data_path, split):
    if cfg['type'] == 'nerf_synthesic':
        return NeRFSynthesicDataset(data_path, split)
    elif cfg['type'] == 'colmap':
        return ColmapDataset(data_path, split)
    else :
        raise NotImplementedError("[ERROR] No such data type: {}.".format(data_path))
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x
