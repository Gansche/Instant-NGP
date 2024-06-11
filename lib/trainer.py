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
    
    def save(self, config, itr):
        net_state = self.model.state_dict()

        ckpt = {
            'config': config,
            'itr': itr,
            'net': net_state,
            'optim': self.optim.state_dict()
        }
        return ckpt
    
    def cuda(self):
        self.model.cuda()
        
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        
    def run(self, epoch, input, batch_size=4096):
        rays = input['rays']
        targets = input['gt_image']
        
        # actually batch processing is meaningless here
        assert rays.shape[0] == 1
        loss_list = []
        for i in range(rays.shape[0]):
            ray = rays[i].cuda()
            target = targets[i][..., :3].cuda()
            
            # random pick some rays
            ray_target = torch.cat((ray, target), dim=-1)
            ray_target = ray_target.reshape(ray_target.shape[0] * ray_target.shape[1], ray_target.shape[2])
            indices = torch.randperm(ray_target.shape[0])[:4096]
            
            ray = ray_target[indices, :6]
            target = ray_target[indices, 6:]
            
            rgb = self.model(ray)
            loss = self.loss(rgb * 255, target)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            decay_rate = 0.1
            decay_steps = 10 * 1000
            new_lrate = 0.01 * (decay_rate ** (epoch / decay_steps))
            for param_group in self.optim.param_groups:
                param_group['lr'] = new_lrate
            
            loss_list.append(loss)
            
        return loss_list

    def render(self, rays, batch_size=4096):
        assert rays.ndim == 4
        assert rays.shape[0] == 1
        rays = rays.squeeze(0)

        H = rays.shape[0]
        W = rays.shape[1]

        rays_reshape = rays.reshape(H * W, 6)
        
        renderings = []
        for i in range(H * W // batch_size + 1):
            with torch.no_grad():
                rendering = self.model(rays_reshape[i * batch_size : (i + 1) * batch_size].cuda())
            renderings.append(rendering.cpu())
        image = torch.cat(renderings, dim=0).reshape(H, W, 3)
        return image
        
        
