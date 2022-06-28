import os
import wget 
import zipfile
import argparse

def create_directory_structure(dest_path, parent_dir):
    final_path = dest_path + "/" + parent_dir
    try:
        os.makedirs(final_path)
    except FileExistsError:
        print(f"{final_path} already exists.")


def download_test_datasets(dest_path):
    urls = [("Set14/", "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip"),]
            # ("BSD100/", "https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip")]
    for directory, url in urls:
        wget.download(url, out=dest_path + "/")


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
    parent_dir = "SR_test"
    create_directory_structure(args.DEST_PATH, args.PARENT_DIR_NAME)
    download_test_datasets(args.DEST_PATH + "/" + args.PARENT_DIR_NAME + "/")
    extract_datasets(args.DEST_PATH + "/" + args.PARENT_DIR_NAME + "/")