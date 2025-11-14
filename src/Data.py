import numpy as np
from PIL import Image
import glob
import os
import pandas as pd
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    A class used to create and manage a dataset out of given data.
    """
    def __init__(self, image_path: str, csv_path: str, pixelsx: int, pixelsy: int) -> None:
        """
        Initializes a dataset out of given data by.

        1. reading a csv file with lables and vectors.
        2. going through the data, which is structured in a folder structure.

        @param image_path: path to the directory which contains all directories that store images.
        @param csv_path: path to the csv-file that contains the labels with vectors.
        @param pixelsx: contains the uniform pixel number all images will get in x axis.
        @param pixelsy: contains the uniform pixel number all images will get in y axis.
        """
        self.image_folderpath = image_path
        self.csv_filepath = csv_path
        self. pixelsx = pixelsx
        self.pixelsy = pixelsy

        self.samples = []
        self.labels = []
        self.string_lables = []
        self.vectors = []

        # Read csv file and extract lables and vectors
        df = pd.read_csv(csv_path, skiprows=0)
        self.labels = df.iloc[:, 0].to_numpy()
        self.string_labels = df.iloc[:, 1].to_numpy()
        self.vectors = df.iloc[:, 2:].to_numpy()
        
        for current_folder in range(43):
            folder_loc = os.path.join(image_path, f"{current_folder:05d}")
            paths = glob.glob(os.path.join(folder_loc, "*.ppm"))

            for p in paths:
                img = Image.open(p).convert("RGB")
                img = img.resize((pixelsx, pixelsy))
                img_array = np.array(img)
                img_array = np.transpose(img_array, (2, 0, 1))
                self.samples.append(img_array)                  


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)


    def __getitem__(self, idx) -> tuple[np.ndarray, int, str, np.ndarray]:
        """
        Returns image, label, string_label and vecor of the dataset at given index.

        @param idx: index of the sample
        """
        image = self.samples[idx]
        label =  self.labels[idx]
        string_label = self.string_labels[idx]
        vector = self.vectors[idx]

        return image, label, string_label, vector