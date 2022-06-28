import os
import wget
import zipfile
import argparse

def create_directory_structure(dest_path, parent_dir):
    child_dirs = ["train", "valid"]
    for directory in child_dirs:
        dir_path = dest_path + "/" + parent_dir + "/" + directory
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            print(f"{dir_path} already exists.")


def download_dataset(dest_path):
    urls = [("train/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"), 
            ("train/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip"), 
            ("train/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip"),
            ("train/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"),
            ("train/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X2.zip"), 
            ("train/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X3.zip"),
            ("train/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip"),
            ("valid/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"),
            ("valid/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip"), 
            ("valid/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip"),
            ("valid/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"),
            ("valid/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X2.zip"), 
            ("valid/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X3.zip"),
            ("valid/", "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip")]
    for directory, url in urls:
        wget.download(url, out=dest_path + "/" + directory)


def extract_datasets(src_path, extension=".zip"):
    for item in os.listdir(src_path):
        if item.endswith(extension):
            zip_path = src_path + "/" + item
            zip_ref = zipfile.ZipFile(zip_path)
            zip_ref.extractall(src_path)
            zip_ref.close()
            os.remove(zip_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DEST_PATH', type=str, required=True)
    parser.add_argument('--PARENT_DIR_NAME', type=str, required=True)
    args = parser.parse_args()
    parent_dir = args.PARENT_DIR_NAME
    create_directory_structure(args.DEST_PATH, args.PARENT_DIR_NAME)
    download_dataset(args.DEST_PATH + "/" + args.PARENT_DIR_NAME)
    extract_datasets(args.DEST_PATH + "/" + args.PARENT_DIR_NAME + "/train")
    extract_datasets(args.DEST_PATH + "/" + args.PARENT_DIR_NAME + "/valid")
