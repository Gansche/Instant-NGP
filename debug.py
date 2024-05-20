import os
import pdb
import yaml
import json
from lib.dataset import make_dataset
from lib.model import InstantNGP

with open('configs/config.yaml','r') as file:
    config = yaml.safe_load(file)
dataset = make_dataset(config['data'], 'data/hotdog')

data = dataset[0]

# rays = data['rays'].view(-1,6)[:1024, ...]
rays = data['rays'].cuda()
model = InstantNGP(config).cuda()

output = model(rays)

pdb.set_trace()