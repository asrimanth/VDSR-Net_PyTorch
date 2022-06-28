import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
import numpy as np

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, images):
        lr_image, hr_image = images['lr_image'], images['hr_image']

        channels, h, w = lr_image.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        lr_image = tvf.crop(lr_image, top, left, new_h, new_w)
        hr_image = tvf.crop(hr_image, top, left, new_h, new_w)

        return {'lr_image': lr_image, 'hr_image': hr_image}


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, images):
        lr_image, hr_image = images['lr_image'], images['hr_image']
        center_crop = transforms.CenterCrop(self.output_size)
        lr_image, hr_image = center_crop(lr_image), center_crop(hr_image)
        return {'lr_image':lr_image, 'hr_image':hr_image}

class MaxNormalize(object):
    def __init__(self):
        pass
    
    def __call__(self, images):
        lr_image, hr_image = images['lr_image'], images['hr_image']
        lr_image, hr_image = lr_image/255.0, hr_image/255.0
        return {'lr_image':lr_image, 'hr_image':hr_image}


class MaxDeNormalize(object):
    def __init__(self):
        pass
    
    def __call__(self, images):
        lr_image, hr_image = images['lr_image'], images['hr_image']
        lr_image, hr_image = lr_image * 255.0, hr_image * 255.0
        return {'lr_image':lr_image, 'hr_image':hr_image}


class RandomRotation(object):
    def __init__(self, angle_range):
        assert isinstance(angle_range, int)
        assert 0 <= angle_range <= 360
        self.angle_range = angle_range
    
    def __call__(self, images):
        lr_image, hr_image = images['lr_image'], images['hr_image']
        angle = np.random.randint(0, self.angle_range)
        lr_image, hr_image = tvf.rotate(lr_image, angle), tvf.rotate(hr_image, angle)
        return {'lr_image':lr_image, 'hr_image':hr_image}


class RandomHorizontalFlip(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        assert 0 < probability < 1
        self.probability = probability
    
    def __call__(self, images):
        lr_image, hr_image = images['lr_image'], images['hr_image']
        if np.random.uniform(0, 1) < self.probability:
            lr_image, hr_image = tvf.hflip(lr_image), tvf.hflip(hr_image)
        return {'lr_image':lr_image, 'hr_image':hr_image}


class RandomVerticalFlip(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        assert 0 < probability < 1
        self.probability = probability
    
    def __call__(self, images):
        lr_image, hr_image = images['lr_image'], images['hr_image']
        if np.random.uniform(0, 1) < self.probability:
            lr_image, hr_image = tvf.vflip(lr_image), tvf.vflip(hr_image)
        return {'lr_image':lr_image, 'hr_image':hr_image}


class Normalize(object):
    def __init__(self, mean, std, inplace=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, images):
        normalize = transforms.Normalize(mean=self.mean, 
                                          std=self.std, 
                                          inplace=self.inplace)
        lr_image, hr_image = images['lr_image'], images['hr_image']
        lr_image, hr_image = normalize(lr_image), normalize(hr_image)
        return {'lr_image':lr_image, 'hr_image':hr_image}


class UnNormalize(object):
    def __init__(self, normal_mean, normal_std, inplace=True):
        self.mean_inv = [-m/s for m,s in zip(normal_mean, normal_std)]
        # self.mean_inv = -1.0 * normal_mean / normal_std
        self.std_inv = [1.0/s for s in list(normal_std)]
        # self.std_inv = 1.0 / normal_std
        self.inplace = inplace
    
    def __call__(self, images):
        normalize = transforms.Normalize(mean=self.mean_inv, 
                                          std=self.std_inv, 
                                          inplace=self.inplace)
        lr_image, hr_image = images['lr_image'], images['hr_image']
        lr_image, hr_image = normalize(lr_image), normalize(hr_image)
        return {'lr_image':lr_image, 'hr_image':hr_image}
