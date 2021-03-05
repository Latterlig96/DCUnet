from data import Dataset
from augmentation import TrainAugmentation
from torch.utils.data import DataLoader
from model import DcUnet
from loss import DiceLoss
from train import Trainer
from config import Config
import torch
import yaml
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config',
                    type=str,
                    default="config.yaml",
                    help="Config file path with predefined parameters to run model")

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config) as config: 
        config = yaml.safe_load(config)
    
    config = Config.load_config_class(config)

    data = Dataset(root_dir=config.train_dir_path,
                   label_dir=config.label_dir_path,
                   transform=TrainAugmentation(config),
                   train_mode=True)
    
    data = DataLoader(dataset=data,
                      batch_size=config.batch_size,
                      shuffle=config.shuffle)
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    loss = DiceLoss()

    model = DcUnet()

    trainer = Trainer(model=model,
                      train_dataset=data,
                      val_dataset=False,
                      loss=loss,
                      epochs=config.epochs,
                      device=device,
                      config=config)

    trainer.train()
