from torchvision import transforms
import torch
from config import Config

class TrainAugmentation:

    def __init__(self,
                 config: Config):
        self.augmentation = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize(config.input_dim),
                                                transforms.ColorJitter(),
                                                transforms.RandomVerticalFlip(p=0.3),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomPerspective(p=0.3),
                                                transforms.ToTensor()])

    def __call__(self, x: torch.Tensor): 
        return self.augmentation(x)

class TestAugmentation: 

    def __init__(self,
                 config: Config): 
        self.augmentation = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize(config.input_dim),
                                                transforms.ToTensor()])
    
    def __call__(self, x: torch.Tensor): 
        return self.augmentation(x)
