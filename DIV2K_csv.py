import pandas as pd
import argparse
import os

def create_csv_from_parent_folder(folder_to_index, dest_path="train.csv"):
    hr_folder, lr_bicubic, _ = sorted([folder_to_index + subfolder + "/" for subfolder in os.listdir(folder_to_index)])
    lr_x2, lr_x3, lr_x4 = [lr_bicubic + subfolder + "/" for subfolder in os.listdir(lr_bicubic)]
    hr_paths = [hr_folder + file for file in os.listdir(hr_folder) if file.endswith(".png")]
    lr_x2_paths = [lr_x2 + file for file in os.listdir(lr_x2) if file.endswith(".png")]
    lr_x3_paths = [lr_x3 + file for file in os.listdir(lr_x3) if file.endswith(".png")]
    lr_x4_paths = [lr_x4 + file for file in os.listdir(lr_x4) if file.endswith(".png")]
    hr_paths = sorted(hr_paths)
    lr_x2_paths = sorted(lr_x2_paths)
    lr_x3_paths = sorted(lr_x3_paths)
    lr_x4_paths = sorted(lr_x4_paths)

    data = {"lr_filename": [], "lr_filepath": [], "scale_factor":[], 
           "hr_filename": [], "hr_filepath": []}
    for i in range(len(hr_paths)):
        data["lr_filename"].append(lr_x2_paths[i].split("/")[-1])
        data["lr_filepath"].append(lr_x2_paths[i])
        data["scale_factor"].append(2)
        data["hr_filename"].append(hr_paths[i].split("/")[-1])
        data["hr_filepath"].append(hr_paths[i])
    
    for i in range(len(hr_paths)):
        data["lr_filename"].append(lr_x3_paths[i].split("/")[-1])
        data["lr_filepath"].append(lr_x3_paths[i])
        data["scale_factor"].append(3)
        data["hr_filename"].append(hr_paths[i].split("/")[-1])
        data["hr_filepath"].append(hr_paths[i])
    
    for i in range(len(hr_paths)):
        data["lr_filename"].append(lr_x4_paths[i].split("/")[-1])
        data["lr_filepath"].append(lr_x4_paths[i])
        data["scale_factor"].append(4)
        data["hr_filename"].append(hr_paths[i].split("/")[-1])
        data["hr_filepath"].append(hr_paths[i])
    data = pd.DataFrame(data)
    data.to_csv(dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--PARENT_DIR_PATH', type=str, required=True)
    args = parser.parse_args()
    parent_folder_path = args.PARENT_DIR_PATH #"/l/vision/v5/sragas/DIV2K/"
    train, valid = [parent_folder_path + subfolder + "/" for subfolder in os.listdir(parent_folder_path)]
    create_csv_from_parent_folder(train, "train_bicubic.csv")
    create_csv_from_parent_folder(valid, "valid_bicubic.csv")
    