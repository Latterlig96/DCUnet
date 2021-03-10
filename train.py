import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from loss import DiceLoss
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

    def build_train_dependencies(self):
        self.build_optimizer() 
        self.build_scheduler()
        self.build_model_cuda()
        
class Trainer(TrainBuilder): 

    logging.getLogger(__name__)

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
        self.model.train() 
        losses = AverageMeter('Loss', ':.4e')
        for epoch in range(self.epochs):
            for index, (images, labels) in enumerate(self.train_dataset):
                images, labels = images.float().to(self.device), labels.float().unsqueeze(1).to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.loss(output, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                losses.update(loss.item(), self.config.batch_size)
                logging.info(losses)

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
