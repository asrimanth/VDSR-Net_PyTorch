import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

from model import *
from dataclass import *
from config_v3 import *
import augmentations

import wandb
# Login only once to wandb before tracking your metrics.
# Once logged in, for a second run, there's no need to relogin.
# wandb.login()

class Trainer:
    """The trainer class, which trains and validates on VDSR network with a given configuration,
        train and validation dataloaders.

        Args:
            train_dataloader (DataLoader): Training dataloader object.
            valid_dataloader (DataLoader): Validation dataloader object.
            config (Object): An object of the class Configuration, which can be modified in config_v3.py
    """
    def __init__(self, train_dataloader, valid_dataloader, config):
        self.config = config
        self.patience = self.config.PATIENCE_EARLY_STOPPING
        self.model = VDSR_Net_v3(depth=self.config.DEPTH)
        self.loss_function = self.config.LOSS_MSE
        self.batch_size = self.config.BATCH_SIZE
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = self.config.DEVICE
        self.epochs = self.config.EPOCHS
        self.lr = self.config.LEARNING_RATE
        self.optim_step_size = self.config.OPTIM_STEP_SIZE
        self.optim_gamma = self.config.OPTIM_GAMMA
        self.grad_clip_max_value = self.config.GRAD_CLIP_MAX_VALUE
        self.momentum=self.config.MOMENTUM
        self.weight_decay=self.config.WEIGHT_DECAY
        self.PSNR = self.config.PSNR
        self.SSIM = self.config.SSIM
        self.device = self.config.DEVICE
        self.val_for_early_stopping = 9999999 # Early stopping
        self.config_wandb = dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            dataset=self.config.DATASET,
            architecture=self.config.ARCHITECTURE
        )
        
        if not os.path.isdir(self.config.MODEL_SAVEPATH):
            os.makedirs(self.config.MODEL_SAVEPATH)
        
        self.log = pd.DataFrame(columns=["model_name", 
                "train_loss", "valid_loss", 
                "train_SSIM", "valid_SSIM", 
                "train_PSNR", "valid_PSNR"])
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.lr, 
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        self.optim_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                      step_size=self.optim_step_size,
                                      gamma=self.optim_gamma)
    

    def get_difference(self, tensor_image_1, tensor_image_2):
        image_1 = tensor_image_1.detach().numpy()
        image_2 = tensor_image_2.detach().numpy()
        difference = image_1 - image_2
        return torch.from_numpy(difference)


    def validation_step(self):
        self.model.eval()
        loss_valid = 0
        psnr_valid = 0
        ssim_valid = 0
        with torch.no_grad():
            for lr_batch, hr_batch, _ in tqdm(self.valid_dataloader, total=len(self.valid_dataloader)):
                lr_batch = lr_batch.to(self.device)
                hr_batch = hr_batch.to(self.device)
                output = self.model(lr_batch)
                loss = self.loss_function(output.data, hr_batch)
                loss_valid += loss.item()
                psnr_valid += self.PSNR(output.data.to("cpu"), hr_batch.cpu())
                ssim_valid += self.SSIM(output.data.to("cpu"), hr_batch.cpu())
        return (psnr_valid/len(self.valid_dataloader), 
                ssim_valid/len(self.valid_dataloader), 
                loss_valid/len(self.valid_dataloader))
    
    
    def is_early_stopping(self, val_loss, epoch):
        if val_loss < self.val_for_early_stopping:
            self.val_for_early_stopping = val_loss
            self.patience = self.config.PATIENCE_EARLY_STOPPING
            print(f"Saving model at Epoch : {epoch}")
            torch.save(self.model.state_dict(), self.config.MODEL_SAVEPATH + "/" +
                        f"{self.config.ARCHITECTURE}_{self.config.BATCH_SIZE}.pth")
            return False
        elif self.patience > 0:
            self.patience -= 1
            return False
        elif self.patience <= 0:
            return True
            
    
    def log_wandb(self, loss_train, loss_valid, psnr_train, psnr_valid, ssim_train, ssim_valid, learning_rate):
        wandb.log({"Training Loss": loss_train, 
                "Validation Loss": loss_valid, 
                "Train " + self.config.PSNR_STR: psnr_train, 
                "Valid " + self.config.PSNR_STR: psnr_valid, 
                "Train " + self.config.SSIM_STR: ssim_train, 
                "Valid " + self.config.SSIM_STR: ssim_valid, 
                "Learning rate": learning_rate})
    
    
    def fit(self):
        run = wandb.init(project=self.config.WANDB_PROJECT_NAME, entity=self.config.WANDB_ENTITY)
        run.save()
        wandb.config = self.config_wandb
        wandb.watch(self.model, log="all", log_freq=10)

        scaler = torch.cuda.amp.GradScaler()
        
        print("-"*25, "THE MODEL BASED ON" , self.config.ARCHITECTURE, "BEGINS TRAINING", "-"*25)
        print(f"NAME OF THE RUN : {run.name}")
        print(f"TRAINING ON {self.config.DEVICE.upper()}")
        
        for epoch in range(self.epochs):
            self.model.train()
            self.model.to(self.device)
            
            psnr_train = 0
            loss_train = 0
            ssim_train = 0
            b_num = 0
            current_learning_rate = self.optim_scheduler.get_lr()[-1]
            print(f"Current learning rate is : {current_learning_rate}")
            
            for lr_batch, hr_batch, _ in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
                lr_batch = lr_batch.to(self.device)
                hr_batch = hr_batch.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    output = self.model(lr_batch)
                    loss = self.loss_function(output, hr_batch)
                
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                # Gradient clipping
                nn.utils.clip_grad_value_(self.model.parameters(), 
                        clip_value=(self.grad_clip_max_value/current_learning_rate))
                scaler.step(self.optimizer)
                scaler.update()
                loss_train += loss.item()
                psnr_train += self.PSNR(output.data.to("cpu"), hr_batch.to("cpu"))
                ssim_train += self.SSIM(output.data.to("cpu"), hr_batch.to("cpu"))
                b_num += 1
            
            self.optim_scheduler.step()
            
            psnr_valid, ssim_valid, loss_valid = self.validation_step()
            psnr_train /= len(self.train_dataloader)
            ssim_train /= len(self.train_dataloader)
            loss_train /= len(self.train_dataloader)
            
            print("-"*10, "STATUS AT EPOCH NO.", epoch, "-"*10)
            print(f"Train PSNR : {psnr_train}, Train SSIM : {ssim_train}, Train loss {loss_train}")
            print(f"Valid PSNR : {psnr_valid}, Valid SSIM : {ssim_valid}, Valid loss {loss_valid}")
            
            self.log.loc[epoch,:] = [f"{self.config.ARCHITECTURE}_{self.config.BATCH_SIZE}.pth", 
                                     f"{loss_train}",
                                     f"{loss_valid}",
                                     f"{ssim_train}",
                                     f"{ssim_valid}",
                                     f"{psnr_train}",
                                     f"{psnr_valid}"]
            self.log.to_csv(self.config.MODEL_SAVEPATH + 
                            f"/{self.config.ARCHITECTURE}_{self.config.BATCH_SIZE}_valid.csv",
                            index=False)
            self.log_wandb(loss_train, loss_valid, psnr_train, psnr_valid, 
                            ssim_train, ssim_valid, current_learning_rate)
            
            if self.is_early_stopping(loss_valid, epoch):
                print(f"Training terminated after {epoch}(s) because of early stopping.")
                return


if __name__ == "__main__":
    config = Configuration()
    print(config)
    train_transforms = transforms.Compose([
        augmentations.RandomCrop(config.IMAGE_SIZE),
        augmentations.RandomRotation(360),
        augmentations.RandomHorizontalFlip(0.5),
        augmentations.RandomVerticalFlip(0.5),
        augmentations.MaxNormalize()
    ])

    valid_transforms = transforms.Compose([
        augmentations.CenterCrop(config.IMAGE_SIZE), 
        augmentations.MaxNormalize()
    ])
    train_dataset = BicubicDataset(config.TRAIN_PATH, image_transforms=train_transforms)
    valid_dataset = BicubicDataset(config.VALID_PATH, image_transforms=valid_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    trainer = Trainer(train_dataloader, valid_dataloader, config)
    trainer.fit()
