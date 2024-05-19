import os
import pdb
import yaml
import json
from lib.dataset import make_dataset

with open('configs/config.yaml','r') as file:
    config = yaml.safe_load(file)
dataset = make_dataset(config['data'], 'data/hotdog')

a = dataset.__getitem__(0)

# with open(os.path.join('data/hotdog', 'transforms_train.json'), 'r') as file:
#     transfroms = json.load(file)

# import imageio
# image = imageio.v2.imread('data/hotdog/train/r_0.png')

pdb.set_trace()