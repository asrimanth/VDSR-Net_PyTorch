import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure


class Configuration:
    WANDB_PROJECT_NAME = "VDSR-DIV2K"
    WANDB_ENTITY = "asrimanth"
    DATASET = "DIV2K"
    TRAIN_PATH = "./train_bicubic_x2.csv"
    VALID_PATH = "./valid_bicubic_x2.csv"
    TEST_PATH = "./set14.csv"
    MODEL_SAVEPATH = "./models/" # Path to save models
    BATCH_SIZE = 64 # Batch size
    DEPTH = 20
    IMAGE_SIZE = (2 * DEPTH) + 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 80
    PATIENCE_EARLY_STOPPING = 20
    LOSS_MSE = nn.MSELoss()
    LOSS_STR = "MSE Loss"
    PSNR = PeakSignalNoiseRatio()
    PSNR_STR = "Peak Signal To Noise Ratio (PSNR)"
    SSIM = StructuralSimilarityIndexMeasure()
    SSIM_STR = "Structural Similarity Index Measure (SSIM)"
    GRAD_CLIP_MAX_VALUE = 1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    LEARNING_RATE = 0.1
    OPTIM_STEP_SIZE = 20
    OPTIM_GAMMA = 0.1
    ARCHITECTURE = "VDSR-Net"
    
    def __str__(self):
        details = ""
        details += "-"*40 + " CONFIGURATION DETAILS " + "-"*40 + "\n"
        details += f"Architecture : {self.ARCHITECTURE}\n"
        details += f"Dataset : {self.DATASET}\n"
        details += f"Batch Size : {str(self.BATCH_SIZE)}\n"
        details += f"Depth of the network : {str(self.DEPTH)}\n"
        details += f"Image patch size : {str(self.IMAGE_SIZE)}\n"
        details += f"Training platform : {self.DEVICE.upper()}\n"
        details += f"Number of epochs : {str(self.EPOCHS)}\n"
        details += f"Gradient clipping with max norm : {str(self.GRAD_CLIP_MAX_VALUE)}\n"
        details += f"Loss Function : {self.LOSS_STR}\n"
        details += f"Performance metric 1 : {self.PSNR_STR}\n"
        details += f"Performance metric 2 : {self.SSIM_STR}\n"
        details += f"Initial Learning rate : {self.LEARNING_RATE}\n"
        details += f"Momentum : {self.MOMENTUM}\n"
        details += f"Weight Decay : {self.WEIGHT_DECAY}\n"
        details += "-"*105
        return details

if __name__ == "__main__":
    print(Configuration())