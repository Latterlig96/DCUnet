from torch.utils.data import Dataset
import pathlib
import os
from glob import glob
from typing import Union, TypeVar
import cv2

AugmentationClass = TypeVar('AugmentationClass')

class Dataset(Dataset): 

    def __init__(self, 
                 root_dir: pathlib.Path,
                 label_dir: Union[bool,pathlib.Path],
                 transform: AugmentationClass,
                 train_mode: bool):
        self.train_mode = train_mode
        self.root_dir = glob(os.path.join(root_dir, '*.tif'))
        self.root_dir = list(map(lambda x: x.replace('\\', '/'), self.root_dir))
        if self.train_mode:
            self.label_dir = glob(os.path.join(label_dir, '*.tif'))
            self.label_dir = list(map(lambda x: x.replace('\\', '/'), self.label_dir))
        self.transform = transform

    def __len__(self): 
        return len(self.root_dir)

    def __getitem__(self, idx: int): 
        image = cv2.imread(self.root_dir[idx])
        if self.transform: 
            image = self.transform(image)
        if self.train_mode: 
            label_image = cv2.imread(self.label_dir[idx])
            return image, label_image
        return image
