import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from config import Config


class TrainAugmentation:

    def __init__(self,
                 config: Config):
        self.augmentation = A.Compose([A.Resize(height=config.input_dim[0], width=config.input_dim[1], p=1),
                                       A.HorizontalFlip(),
                                       A.VerticalFlip(),
                                       A.RandomBrightnessContrast(),
                                       ToTensorV2()], p=1)

    def __call__(self, x: np.ndarray, mask: np.ndarray): 
        transform = self.augmentation(image=x, mask=mask)
        return transform['image'], transform['mask']

class TestAugmentation: 

    def __init__(self, 
                 config: Config):
        self.augmentation = A.Compose([A.Resize(height=config.input_dim[0], width=config.input_dim[1]),
                                       ToTensorV2()], p=1)
    
    def __call__(self, x: np.ndarray, mask: np.ndarray):
        transform = self.augmentation(image=x, mask=mask)
        return transform['image'], transform['mask']
