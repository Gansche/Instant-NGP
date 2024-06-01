import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
        
    def forward(self, rgb, target):    
        image_loss = torch.mean(torch.pow((rgb - target), 2))
        
        loss = image_loss
        return loss