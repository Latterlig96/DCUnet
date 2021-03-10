from torch.utils.data import Dataset
import pathlib
from typing import Union, TypeVar, List
import cv2

AugmentationClass = TypeVar('AugmentationClass')

class Dataset(Dataset): 

    def __init__(self, 
                 root_dir: pathlib.Path,
                 label_dir: Union[bool,pathlib.Path],
                 transform: List[AugmentationClass]):
        self.root_dir = list(map(lambda x: x.replace('\\', '/'), root_dir))
        self.label_dir = list(map(lambda x: x.replace('\\', '/'), label_dir))
        self.transform = transform

    def __len__(self): 
        return len(self.root_dir)

    def __getitem__(self, idx: int): 
        image = cv2.imread(self.root_dir[idx], 1)
        mask = cv2.imread(self.label_dir[idx], 0)
        if self.transform:
            image, mask = self.transform(image, mask)
        image, mask = image.transpose((2, 0, 1)) / 255.0, mask / 255.0

        return image, mask
