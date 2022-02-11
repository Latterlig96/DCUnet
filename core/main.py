import argparse
import logging
from glob import glob
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from augmentation import TestAugmentation, TrainAugmentation
from config import Config
from dataset import Dataset
from loss import FocalTverskyLoss
from model import DcUnet
from train import Trainer
from utils import seed_everything

parser = argparse.ArgumentParser()

parser.add_argument('--config',
                    type=str,
                    default="config.yaml",
                    help="Config file path with predefined parameters to run model")

if __name__ == "__main__":
    args = parser.parse_args()

    config = Config.load_config_class(args.config)

    samples = glob(config.train_dir_path + "*.tif")
    masks = glob(config.label_dir_path + "*.tif")

    logging.basicConfig(level=config.level,
                        format='%(asctime)s %(levelname)-8s %(name)-15s %(message)s')

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    seed_everything(config.random_state)

    x_train, x_val, y_train, y_val = train_test_split(samples,
                                                      masks,
                                                      test_size=config.test_size,
                                                      random_state=config.random_state)

    train_data = Dataset(root_dir=x_train,
                         label_dir=y_train,
                         transform=TrainAugmentation(config))

    val_data = Dataset(root_dir=x_val,
                       label_dir=y_val,
                       transform=TestAugmentation(config))

    train_data = DataLoader(dataset=train_data,
                            batch_size=config.batch_size,
                            shuffle=config.shuffle)

    val_data = DataLoader(dataset=val_data,
                          batch_size=config.batch_size,
                          shuffle=config.shuffle)

    loss = FocalTverskyLoss()

    model = DcUnet(input_channels=config.num_channels)

    trainer = Trainer(model=model,
                      train_dataset=train_data,
                      val_dataset=val_data,
                      loss=loss,
                      epochs=config.epochs,
                      device=device,
                      config=config)

    trainer.train(save_model_path=config.save_model_path, visualize=True)
