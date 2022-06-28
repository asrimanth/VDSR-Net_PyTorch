import pandas as pd

def create_subset(csv_path, scale_factor, dest_file_path):
    data = pd.read_csv(csv_path)
    data.drop("Unnamed: 0", axis=1, inplace=True)
    subset = data[data["scale_factor"] == scale_factor]
    subset.to_csv(dest_file_path)

if __name__ == "__main__":
    create_subset("./train_bicubic.csv", 2, "./train_bicubic_x2.csv")
    create_subset("./valid_bicubic.csv", 2, "./valid_bicubic_x2.csv")