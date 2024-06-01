import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class InstantNGPTrainer():
    def __init__(self, optim, model, loss) -> None:
        self.optim = optim
        self.model = model 
        self.loss = loss
        
    def load(self, path):
        pass
    
    def save(self, path):
        pass
    
    def cuda(self):
        self.model.cuda()
        
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        
    def forward(self, input):
        rays = input['rays']
        targets = input['gt_image']
        
        # actually batch processing is meaningless here
        for i in range(rays.shape[0]):
            ray = rays[i].cuda()
            target = targets[i].cuda()
            
            # random pick some rays
            ray = ray[0, :]
            target = target[0, :, :3]
            
            rgb = self.model(ray)
            loss = self.loss(rgb, target)
            
            loss.backward()
            self.optim.step()

    def render(self, rays):
        return self.model(rays)
        
        
