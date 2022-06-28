import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import pandas as pd


class BicubicDataset(Dataset):
    def __init__(self, csv_path, image_transforms=None):
        self.data = pd.read_csv(csv_path)
        self.image_transforms = image_transforms
    
    def __len__(self):
        return self.data.shape[0]
    
    def get_difference(self, tensor_image_1, tensor_image_2):
        image_1 = tensor_image_1.detach().numpy()
        image_2 = tensor_image_2.detach().numpy()

        difference = image_1 - image_2
        return torch.from_numpy(difference)
    
    def __getitem__(self, index):
        lr_image = Image.open(self.data.iloc[index, 2]).convert("RGB")
        scale_factor = int(self.data.iloc[index, 3])
        hr_image = Image.open(self.data.iloc[index, 5]).convert("RGB")
        lr_width, lr_height = lr_image.size
        lr_image = lr_image.resize((lr_width * scale_factor, lr_height * scale_factor), 
                    resample=Image.Resampling.BICUBIC)
        pil_to_tensor = transforms.PILToTensor()

        lr_image = pil_to_tensor(lr_image)
        hr_image = pil_to_tensor(hr_image)
        lr_image = lr_image.to(torch.float)
        hr_image = hr_image.to(torch.float)

        if self.image_transforms is not None:
            sample = {'lr_image': lr_image, 'hr_image': hr_image}
            out_sample = self.image_transforms(sample)
            lr_image, hr_image = out_sample['lr_image'], out_sample['hr_image']
        return lr_image, hr_image, self.get_difference(lr_image, hr_image)


class StandardDataset(Dataset):
    def __init__(self, csv_path, image_transforms=None):
        self.data = pd.read_csv(csv_path)
        self.image_transforms = image_transforms
    
    def __len__(self):
        return self.data.shape[0]
    
    def get_difference(self, tensor_image_1, tensor_image_2):
        image_1 = tensor_image_1.detach().numpy()
        image_2 = tensor_image_2.detach().numpy()

        difference = image_1 - image_2
        return torch.from_numpy(difference)
    
    def __getitem__(self, index):
        lr_image = Image.open(self.data.iloc[index, 2]).convert("RGB")
        scale_factor = int(self.data.iloc[index, 3])
        hr_image = Image.open(self.data.iloc[index, 5]).convert("RGB")
        interpolation_methods = [Image.Resampling.NEAREST, Image.Resampling.BOX, 
                                Image.Resampling.BILINEAR, Image.Resampling.HAMMING, 
                                Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]
        rand_int = np.random.randint(0, high=len(interpolation_methods))
        lr_width, lr_height = lr_image.size
        lr_image = lr_image.resize((lr_width * scale_factor, lr_height * scale_factor), 
                    resample=interpolation_methods[rand_int])
        pil_to_tensor = transforms.PILToTensor()

        lr_image = pil_to_tensor(lr_image)
        hr_image = pil_to_tensor(hr_image)
        lr_image = lr_image.to(torch.float)
        hr_image = hr_image.to(torch.float)

        if self.image_transforms is not None:
            sample = {'lr_image': lr_image, 'hr_image': hr_image}
            out_sample = self.image_transforms(sample)
            lr_image, hr_image = out_sample['lr_image'], out_sample['hr_image']
        return lr_image, hr_image, self.get_difference(lr_image, hr_image)


class TestDataset:
    def __init__(self, csv_path, image_transforms=None):
        self.data = pd.read_csv(csv_path)
        self.image_transforms = image_transforms

    def __len__(self):
        return self.data.shape[0]
    
    def get_difference(self, tensor_image_1, tensor_image_2):
        image_1 = tensor_image_1.detach().numpy()
        image_2 = tensor_image_2.detach().numpy()

        difference = image_1 - image_2
        return torch.from_numpy(difference)
    
    def __getitem__(self, index):
        lr_image = Image.open(self.data.iloc[index, 2]).convert("RGB")
        hr_image = Image.open(self.data.iloc[index, 5]).convert("RGB")

        pil_to_tensor = transforms.PILToTensor()
        lr_image = pil_to_tensor(lr_image)
        hr_image = pil_to_tensor(hr_image)
        lr_image = lr_image.to(torch.float)
        hr_image = hr_image.to(torch.float)

        if self.image_transforms is not None:
            sample = {'lr_image': lr_image, 'hr_image': hr_image}
            out_sample = self.image_transforms(sample)
            lr_image, hr_image = out_sample['lr_image'], out_sample['hr_image']
        
        return lr_image, hr_image, self.get_difference(lr_image, hr_image)
