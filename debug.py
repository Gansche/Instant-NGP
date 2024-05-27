import os
import pdb
import yaml
import json
from lib.dataset import make_dataset
from lib.model import InstantNGP

import torch

with open('configs/config.yaml','r') as file:
    config = yaml.safe_load(file)
dataset = make_dataset(config['data'], 'data/chair')

data = dataset[0]

bb = dataset.create_bounding_box().cuda()

rays = data['rays'].view(-1,6)[:1024, ...].cuda()
model = InstantNGP(config, bb).cuda()

output = model(rays)
pdb.set_trace()
