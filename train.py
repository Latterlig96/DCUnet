import torch
import torch.nn as nn 
from typing import Optional, TypeVar
from torch.utils.data import DataLoader
from config import Config
from utils import AverageMeter
import logging 

class TrainBuilder:
    
    def __init__(self):
        self.build_train_dependencies()

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config.learning_rate,
                                     betas=self.config.betas)
    
    def build_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                               T_max=self.config.T_max,
                                                               eta_min=self.config.eta_min)
    
    def build_model_cuda(self):
        self.model.to(self.device)
    
    def build_scaler(self):
        self.scaler = torch.cuda.amp.GradScaler()
    
    def build_train_dependencies(self):
        self.build_optimizer() 
        self.build_scheduler()
        self.build_model_cuda()
        self.build_scaler()
        
class Trainer(TrainBuilder): 

    logging.getLogger(__name__)

    def __init__(self,
                 model: nn.Module,
                 train_dataset: DataLoader,
                 val_dataset: DataLoader,
                 loss: TypeVar("loss_instance"),
                 epochs: int,
                 device: str,
                 config: Config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss = loss
        self.epochs = epochs
        self.device = device
        self.config = config
        super(Trainer, self).__init__()

    def train(self,
              save_model_path: Optional[str]=None,
              eval_after_epoch: bool=True): 
        losses = AverageMeter('Loss', ':.4e')
        min_loss = float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            for index, (images, labels) in enumerate(self.train_dataset):
                images, labels = images.float().to(self.device), labels.float().unsqueeze(1).to(self.device)
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = self.model(images)
                    loss = self.loss(output, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
                losses.update(loss.item(), self.config.batch_size)
                logging.info(losses)
            if eval_after_epoch:
                logging.info(f"Epoch {epoch} end, running inference mode on validation dataset")
                avg_loss = self.eval()
                logging.info("Done validation step")
        
                if save_model_path and avg_loss < min_loss: 
                    logging.info(f"Saving model to {save_model_path}")
                    torch.save(self.model.state_dict(), save_model_path)
                    logging.info("Model saved successfully")
                    min_loss = avg_loss

        if save_model_path and self.val_dataset is None:
            logging.info(f"Saving model to {save_model_path}")
            torch.save(self.model.state_dict(), save_model_path)
            logging.info("Model saved successfully")

    def eval(self):
        with torch.no_grad(): 
            losses = AverageMeter('Loss', ':.4e')
            self.model.eval()
            for index, (images, labels) in enumerate(self.val_dataset): 
                self.optimizer.zero_grad()
                images, labels = images.float().to(self.device), labels.float().unsqueeze(1).to(self.device)
                output = self.model(images)
                loss = self.loss(output, labels)
                losses.update(loss.item(), self.config.batch_size)
                logging.info(losses)
            return losses.get_avg_loss()
