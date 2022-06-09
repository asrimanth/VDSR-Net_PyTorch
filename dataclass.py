import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as tvf
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from PIL import Image


class StandardDataset(Dataset):
    def __init__(self, csv_path, is_transform=False):
        self.data = pd.read_csv(csv_path)
        self.is_transform = is_transform
    
    def __len__(self):
        return self.data.shape[0]

    def get_difference(self, tensor_image_1, tensor_image_2):
        image_1 = tensor_image_1.detach().numpy()
        image_2 = tensor_image_2.detach().numpy()

        difference = image_1 - image_2

        return torch.from_numpy(difference)
    
    def __getitem__(self, index):
        lr_image = read_image(self.data.iloc[index, 2])
        lr_height, lr_width = tvf.get_image_size(lr_image)
        resize = transforms.Resize((lr_width*2, lr_height*2))
        
        # Normalization using ImageNet measures of center and spread.
        normalize = transforms.Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), 
                                          std=torch.Tensor([0.229, 0.224, 0.225]), 
                                          inplace=True)
        # tensorify = transforms.ToTensor()
        lr_interpolated_image = resize(lr_image)
        hr_image = read_image(self.data.iloc[index, 5])
        if self.is_transform:
            if random.random() > 0.5:
                angle = random.randint(0, 180)
                lr_interpolated_image = tvf.rotate(lr_interpolated_image, angle)
                hr_image = tvf.rotate(hr_image, angle)
                
            if random.random() > 0.5:
                lr_interpolated_image = tvf.hflip(lr_interpolated_image)
                hr_image = tvf.hflip(hr_image)
            
            if random.random() > 0.5:
                lr_interpolated_image = tvf.vflip(lr_interpolated_image)
                hr_image = tvf.vflip(hr_image)
        
        lr_interpolated_image = normalize(lr_interpolated_image.type(torch.float32))
        hr_image = normalize(hr_image.type(torch.float32))
        return lr_interpolated_image, hr_image, self.get_difference(hr_image, lr_interpolated_image).type(torch.float32)