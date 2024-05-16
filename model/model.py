import torch.nn as nn

class InstantNGP(nn.Module):
    def __init__(self) -> None:
        super(InstantNGP, self).__init__()
        
        self.encoder = HashEncoder()
        self.nerf = NeRF()
        
    def forward(self):
        pass

class HashEncoder(nn.Module):
    def __init__(self) -> None:
        super(HashEncoder, self).__init__()
        
    def forward(self):
        pass
    
class NeRF(nn.Module):
    def __init__(self) -> None:
        super(NeRF, self).__init__()
        
    def forward(self):
        pass