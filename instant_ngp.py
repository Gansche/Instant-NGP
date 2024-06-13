import os
import yaml
import random
import imageio
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter

from lib.utils import *
from lib.dataset import make_dataset, cycle
from lib.trainer import InstantNGPTrainer
from lib.model import InstantNGP
from lib.loss import Loss
from lib.radam import RAdam
    
import pdb

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
    
def training(config, args):
    """ seetings """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    writer = SummaryWriter(config['tensorboard_path'])
    
    ###########################################################################
    """ dataset """
    dataset =  make_dataset(config['data'], args.data_path)
    bb = dataset.create_bounding_box().cuda()
    
    # split dataset
    pivot = int(len(dataset) * 0.95)
    train_data, valid_data = random_split(dataset, [pivot, len(dataset) - pivot])
    
    # train dataset
    train_loader = DataLoader(
        train_data, batch_size=config['data']['batch_size'],
        shuffle=True, drop_last=False
    )
    train_iterator = cycle(train_loader)
    
    # validation dataset
    valid_loader = DataLoader(
        valid_data, batch_size=config['data']['batch_size'],
        shuffle=False, drop_last=False
    )
    valid_iterator = cycle(valid_loader)
    
    ###########################################################################
    """ preparation """
    tensorboard_path = os.path.abspath(config['tensorboard_path'])
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    checkpoints_path = os.path.abspath(config['checkpoints_path'])
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    
    model = InstantNGP(config, bb)
    loss = Loss()
    optim = RAdam(
        [
            {'params': model.pos_enc.parameters(), 'weight_decay': 1e-6},
            {'params': model.decoder.parameters(), 'eps': 1e-15}
        ], lr=0.01, betas=(0.9, 0.99)
    )
    trainer = InstantNGPTrainer(optim, model, loss)
    
    trainer.cuda()
    
    ###########################################################################
    """ train & validation """
    for epoch in tqdm(range(config['epoches']), desc='Training', unit='epoch'):
        trainer.train()
        data = next(train_iterator)
        
        loss_list = trainer.run(epoch, data)
        assert len(loss_list) == 1
        
        writer.add_scalars('Loss', {'Training': loss_list[0]}, epoch)

        if epoch % config['valid_iter'] == 0 and epoch != 0:
            trainer.eval()
            folder_name = 'renderings/epoch_{0:05d}'.format(epoch)
            if os.path.exists(folder_name):
                os.rmdir(folder_name)
            os.mkdir(folder_name)

            for pose, valid_data in enumerate(valid_loader):
                gt_image = valid_data['gt_image'].squeeze(0)
                pred_image = trainer.render(valid_data['rays'])
                pred_image = to8b(pred_image.numpy())
                imageio.imwrite(folder_name + '/gt_image' + '_{0:05d}'.format(pose) + '.png', (gt_image.numpy()).astype(np.uint8))
                # imageio.imwrite('renderings/epoch_{0:05d}/'.format(epoch) + 'pred_image' + '_{0:05d}'.format(pose) + '.png', (pred_image.numpy() * 255).astype(np.uint8))
                imageio.imwrite(folder_name + '/pred_image' + '_{0:05d}'.format(pose) + '.png', (pred_image).astype(np.uint8))
                      
        if epoch % config['save_iter'] == 0 and epoch != 0:
            ckpt = trainer.save(config, epoch)
            torch.save(ckpt, os.path.join(checkpoints_path, 'ckpt_' + '{0:05d}_epoch'.format(epoch) + '.pth'))
        

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