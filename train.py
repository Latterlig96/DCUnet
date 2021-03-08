import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from loss import DiceLoss
from config import Config
from utils import AverageMeter


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
        
    def build_train_dependencies(self):
        self.build_optimizer() 
        self.build_scheduler()

class Trainer(TrainBuilder): 

    def __init__(self,
                 model: nn.Module,
                 train_dataset: DataLoader,
                 val_dataset: DataLoader,
                 loss: DiceLoss,
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

    def train(self):
        self.model.to(self.device)
        self.model.train() 
        losses = AverageMeter('Loss', ':.4e')
        for epoch in range(self.epochs):
            for images, labels in self.train_dataset: 
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.loss(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), self.config.batch_size)
                print(losses)

    def eval(self):
        with torch.no_grad(): 
            losses = AverageMeter('Loss', ':.4e')
            self.model.eval()
            for images, labels in self.val_dataset: 
                self.optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.loss(output, labels)
                losses.update(loss.item(), self.config.batch_size)
                print(losses)
