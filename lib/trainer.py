import torch
import torch.nn as nn
import torch.nn.functional as F

class InstantNGPTrainer():
    def __init__(self, optim, sched, model, loss) -> None:
        self.optim = optim
        self.sched = sched
        self.model = model 
        self.loss = loss
        
    def load(self):
        pass
    
    def save(self):
        pass
    
    def cuda(self):
        pass
        
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        
    def forward(self):
        pass
