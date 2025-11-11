import numpy as np
from PIL import Image
import glob
import os
import pandas as pd
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    
    def __init__(self, image_folderpath: str, csv_filepath: str, pixelsx: int, pixelsy: int):
        self.image_folderpath = image_folderpath
        self.csv_filepath = csv_filepath
        self. pixelsx = pixelsx
        self.pixelsy = pixelsy

        self.samples = []
        self.labels = []
        self.string_lables = []
        self.vectors = []

        # Read csv file and extract lables and vectors
        dataset = []
        df = pd.read_csv(csv_filepath, skiprows=0)
        self.labels = df.iloc[:, 0].to_numpy()
        self.string_labels = df.iloc[:, 1].to_numpy()
        self.vectors = df.iloc[:, 2:].to_numpy()


        # Read image samples
        current_folder = 0

        for current_folder in range(42):
            folder_loc = os.path.join(image_folderpath, f"{current_folder:05d}")
            paths = sorted(glob.glob(os.path.join(folder_loc, "*.ppm")))

            for p in paths:
                img = Image.open(p).convert("RGB")

                # images have different sizes so they have to be resized
                img = img.resize((pixelsx, pixelsy))
                img_array = np.array(img)
                img_array = np.transpose(img_array, (2, 0, 1))
                self.samples.append(img_array)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        image = self.samples[idx]
        label =  self.labels[idx]
        string_label = self.string_labels[idx]
        vector = self.vectors[idx]

        return image, label, string_label, vector


if __name__ == "__main__":
    folderpath = "../data/raw/GTSRB/Final_Training/Images/"
    filepath = "../data/raw/concepts_per_class.csv"
    dataset = ImageDataset(folderpath, filepath, 128, 128)
    print("dataset created")
    print(dataset.__getitem__(1))
