import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio

class Configuration:
    TRAIN_PATH = "./DIV2K_train_subset.csv"
    VALID_PATH = "./DIV2K_valid_subset.csv"
    MODEL_SAVEPATH = "./models/" #path to save models
    BATCH_SIZE = 4 #batch size
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 3
    DEPTH = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 80
    LOSS_MSE = nn.MSELoss()
    LOSS_MSE_STR = "MSE Loss"
    LOSS_CXE = nn.CrossEntropyLoss()
    LOSS_CXE_STR = "CXE Loss"
    EVALUATION_METRIC = PeakSignalNoiseRatio()
    EVALUATION_STR = "Peak Signal To Noise Ratio (PSNR)"
    GRAD_CLIP_MAX_NORM = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 1e-4
    OPTIM_STEP_SIZE = 20
    OPTIM_GAMMA = 0.1
    ARCHITECTURE = "VDSR-Net"
    
    def __str__(self):
        details = ""
        details += "-"*40 + " CONFIGURATION DETAILS " + "-"*40 + "\n"
        details += f"Architecture : {self.ARCHITECTURE}\n"
        details += f"Batch Size : {str(self.BATCH_SIZE)}\n"
        details += f"Number of Input Channels : {str(self.INPUT_CHANNELS)}\n"
        details += f"Number of Output Channels : {str(self.OUTPUT_CHANNELS)}\n"
        details += f"Depth of the network : {str(self.DEPTH)}\n"
        details += f"Training platform : {self.DEVICE.upper()}\n"
        details += f"Number of epochs : {str(self.EPOCHS)}\n"
        details += f"Gradient clipping with max norm : {str(self.GRAD_CLIP_MAX_NORM)}\n"
        details += f"Loss Function : {self.LOSS_CXE_STR}\n"
        details += f"Performance metric : {self.EVALUATION_STR}\n"
        details += "-"*106
        return details