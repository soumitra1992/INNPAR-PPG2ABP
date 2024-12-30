import torch
from torch import nn

class model_loss(nn.Module):
    def __init__(self,reduction='mean'):
        super(model_loss, self).__init__()
        self.criterion = nn.L1Loss(reduction=reduction)
        
    def forward(self, pred, target):
        loss=self.criterion(pred, target)
        return loss