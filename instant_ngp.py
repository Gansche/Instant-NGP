import os
import yaml
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from lib.utils import *
from lib.dataset import make_dataset, cycle
    
import pdb
    
def training(config, args):
    """ seetings """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    writer = SummaryWriter(config['ckpt_path'])
    
    ###########################################################################
    """ dataset """
    dataset =  make_dataset(config['data'], args.data_path)
    
    # split dataset
    pivot = int(len(dataset) * 0.8)
    train_data, valid_data = random_split(dataset, [pivot, len(dataset) - pivot])
    
    # TODO : collect function?
    train_loader = DataLoader(
        train_data, batch_size=config['data']['batch_size'],
        shuffle=True, drop_last=False
    )
    train_iterator = cycle(train_loader)
    
    valid_loader = DataLoader(
        valid_data, batch_size=config['data']['batch_size'],
        shuffle=True, drop_last=False
    )
    valid_iterator = cycle(valid_loader)
    
    ###########################################################################
    """ trainer """
    # trainer = make_stylization_trainer(config)
    
    ###########################################################################
    """ train & validation """
    for epoch in tqdm(range(5000)):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training stylizer")
    parser.add_argument('-c', '--config_path', type=str, default='configs/config.yaml', help='config file path')
    parser.add_argument('-d', '--data_path', type=str, default='data/hotdog', help='config file path')
    args = parser.parse_args()
    
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.load(f, yaml.FullLoader)
    except:
        raise FileNotFoundError('[ERROR] model loading failed: {:s}'.format(args.config_path))
    
    training(config, args)
    print("Training complete.")