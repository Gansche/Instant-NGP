import torch
import torch.nn as nn
import torch.nn.functional as F

class InstantNGPTrainer():
    def __init__(self, optim, sched, model) -> None:
        self.optim = optim
        self.sched = sched
        self.model = model 
    
def make_optim(cfg):
    pass

def make_sched(cfg):
    pass

def make_trainer(cfg, model, loss):
    optim = make_optim(cfg)
    sched = make_sched(cfg)
    trainer = InstantNGPTrainer(optim, sched, model)
    return trainer
