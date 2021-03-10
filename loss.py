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

class FocalLoss(nn.Module):
    def __init__(self, 
                 alpha: float=0.25, 
                 gamma: int=2, 
                 weight: float=None, 
                 ignore_index: int=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, 
                inputs: torch.Tensor,  
                targets: torch.Tensor
                ) -> torch.Tensor:
                
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            targets = targets[mask]
            inputs = inputs[mask]

        logpt = -self.bce_fn(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

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
