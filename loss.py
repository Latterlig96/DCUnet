import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                smooth: int=1):       
        inputs = F.sigmoid(inputs)   
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
        inputs = F.sigmoid(inputs)    
        inputs = inputs.view(-1)
        targets = targets.view(-1)        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class TverskyLoss(nn.Module):

    def __init__(self): 
        super(TverskyLoss, self).__init__()
    
    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor, 
                alpha: float = 0.75,
                smooth: int = 1,
                gamma: float = 0.75) -> torch.Tensor:
        inputs = F.sigmoid(inputs)
        output = self.tversky(inputs, targets,
                              alpha, smooth)
        return torch.pow((1-output), gamma)

    def tversky(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                alpha: float,
                smooth: int
                ) -> torch.Tensor:
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        true_pos = torch.sum(inputs*targets)
        false_neg = torch.sum(inputs*(1-targets))
        false_pos = torch.sum((1-inputs)*targets)
        return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
