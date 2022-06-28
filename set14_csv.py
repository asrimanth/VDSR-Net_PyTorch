import numpy as np
import pandas as pd
import os

def is_lr_interpolated_image(path):
    if not path.endswith("HR.png") and not path.endswith("LR.png") and path.endswith(".png"):
        return True
    return False

def create_csv_from_directory(test_path, dest_path="set14.csv"):
    lr_paths = [test_path + file for file in os.listdir(test_path) if is_lr_interpolated_image(test_path + file)]
    hr_paths = [test_path + file for file in os.listdir(test_path) if file.endswith("HR.png")]
    lr_paths = sorted(lr_paths)
    hr_paths = sorted(hr_paths)
    data = {"lr_filename": [], "lr_filepath": [], "lr_method": [], 
           "hr_filename": [], "hr_filepath": []}
    
    x = 0
    for i, hr_path in enumerate(hr_paths):
        for j in range((len(lr_paths) // len(hr_paths))):
            lr_index = x * 7 + j
            lr_path = lr_paths[lr_index]
            hr_name = hr_path.split("/")[-1]
            lr_name = lr_path.split("/")[-1]
            lr_interpolation_method = lr_name.split(".png")[0].split("_")[-1]
            data["lr_filename"].append(lr_name)
            data["lr_filepath"].append(lr_path)
            data["lr_method"].append(lr_interpolation_method)
            data["hr_filename"].append(hr_name)
            data["hr_filepath"].append(hr_path)
        x += 1
    data = pd.DataFrame(data)
    data.to_csv(dest_path)


if __name__ == "__main__":
    create_csv_from_directory("/l/vision/v5/sragas/SR_test/Set14/image_SRF_2/")
