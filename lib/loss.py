import torch
import torch.nn as nn

class PictureLoss(nn.Module):
    def __init__(self) -> None:
        super(PictureLoss, self).__init__()