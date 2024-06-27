import os
import pdb
import yaml
import json
from lib.dataset import make_dataset, FoxDataset
from lib.model import InstantNGP
import random

import torch
import argparse
import imageio
from PIL import Image 

# random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

# parser = argparse.ArgumentParser(description="Training stylizer")
# parser.add_argument('-c', '--config_path', type=str, default='configs/config.yaml', help='config file path')
# parser.add_argument('-d', '--data_path', type=str, default='fox', help='config file path')
# args = parser.parse_args()

# try:
#     with open(args.config_path, 'r') as f:
#         config = yaml.load(f, yaml.FullLoader)
# except:
#     raise FileNotFoundError('[ERROR] model loading failed: {:s}'.format(args.config_path))

# dataset = FoxDataset('data/' + args.data_path, config['data'])
# bb = dataset.create_bounding_box()

# a = dataset[0]
filename = 'output.mp4'
filepath = os.path.join(os.getcwd(), filename)
root_dir = 'renderings/hotdog' 

images = []

for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)

    if os.path.isdir(folder_path) and 'gt_image' not in os.path.basename(folder_path):
        file_name = os.path.join(folder_path, "00000.png")
        images.append(file_name)

fps = 10
with imageio.get_writer(filepath, fps=fps) as video:
    for image in images:
        frame = imageio.imread(image)
        video.append_data(frame)
