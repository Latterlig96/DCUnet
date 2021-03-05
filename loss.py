import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                smooth: int=1): 
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class IoULoss(nn.Module):

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                smooth: int=1):       
        inputs = inputs.view(-1)
        targets = targets.view(-1)        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
