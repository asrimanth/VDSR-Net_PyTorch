import torch
import torchvision.transforms as transforms
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from torchvision.io import read_image
import torchvision.transforms.functional as tvf
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from PIL import Image

import augmentations
from config_v3 import *
from model import *
from dataclass import TestDataset


def get_difference(tensor_image_1, tensor_image_2):
        image_1 = tensor_image_1.detach().numpy()
        image_2 = tensor_image_2.detach().numpy()

        difference = image_1 - image_2
        return torch.from_numpy(difference)

# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
def save_current_prediction(images_list, dest_path):
    widths, heights = zip(*(i.size for i in images_list))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images_list:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(dest_path)
    new_im.close()



def test_report(model, test_dataset, config):
    psnr_avg_test = 0
    psnr_avg_interpolation = 0
    ssim_avg_test = 0
    ssim_avg_interpolation = 0
    index = 0

    for low_res_image, gt_tensor, residual in tqdm(test_dataset):
        low_res_image =  low_res_image.cpu()
        gt_tensor = gt_tensor.cpu()
        residual = residual.cpu()
        output = model(low_res_image.unsqueeze(0).to(config.DEVICE))
        output = output.cpu().squeeze(0)

        psnr_avg_test += config.PSNR(output.data.cpu(), gt_tensor.cpu())
        psnr_avg_interpolation += config.PSNR(low_res_image.cpu(), gt_tensor.cpu())
        ssim_avg_test += config.SSIM(output.unsqueeze(0).data.cpu(), gt_tensor.unsqueeze(0).cpu())
        ssim_avg_interpolation += config.SSIM(low_res_image.unsqueeze(0).cpu(), gt_tensor.unsqueeze(0).cpu())

        images_list = [low_res_image, output, gt_tensor, 
                            get_difference(low_res_image, output), 
                            get_difference(low_res_image, gt_tensor)]
        images_list = list(map(tvf.to_pil_image, images_list))
        save_current_prediction(images_list, f"./set14_results/pred_{index}.png")
        index += 1

    psnr_avg_test /= len(test_dataset)
    psnr_avg_interpolation /= len(test_dataset)
    ssim_avg_test /= len(test_dataset)
    ssim_avg_interpolation /= len(test_dataset)
    return [("PSNR from the model", psnr_avg_test), 
            ("PSNR using interpolation", psnr_avg_interpolation), 
            ("SSIM from the model", ssim_avg_test), 
            ("SSIM using interpolation", ssim_avg_interpolation)]


if __name__ == "__main__":
    config = Configuration()
    model = VDSR_Net_v3()
    model.load_state_dict(torch.load("./models/VDSR-Net_64_v3_0.pth"))
    model.eval()
    model.to(config.DEVICE)

    test_transforms = transforms.Compose([
        augmentations.CenterCrop(config.IMAGE_SIZE), 
        augmentations.MaxNormalize()
    ])

    test_dataset = TestDataset(config.TEST_PATH, test_transforms)
    print(test_report(model, test_dataset, config))
