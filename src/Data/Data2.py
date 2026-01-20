import numpy as np
from PIL import Image
import glob
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class ImageDataset2(Dataset):
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
        self.pixelsx = pixelsx
        self.pixelsy = pixelsy

        #self.samples = []
        self.sample_paths = []
        self.labels = []
        self.string_labels = []
        self.vectors = []

        # Read csv file and extract names and vectors
        df = pd.read_csv(csv_path, skiprows=0)
        self.string_labels = df.iloc[:, 1].to_numpy()
        self.vectors = df.iloc[:, 2:].to_numpy()

        for current_label in range(43):
            folder_loc = os.path.join(image_path, f"{current_label:05d}")

            paths = glob.glob(os.path.join(folder_loc, "*.ppm"))
            for p in paths:
                self.sample_paths.append(p)
                self.labels.append(current_label)


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.sample_paths)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns image, label, string_label and vector of the dataset at given index.

        @param idx: index of the sample
        """
        path = self.sample_paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.pixelsx, self.pixelsy))
        img_array = np.array(img)
        img_array = np.transpose(img_array, (2, 0, 1))



        # sample = self.samples[idx]
        label =  self.labels[idx]
        vector = self.vectors[label]

        # Transform everything to tensors (needed for ML usage)
        img_array = torch.tensor(img_array, dtype=torch.float32) /255.0
        label = torch.tensor(label, dtype=torch.long)
        vector = torch.tensor(vector, dtype=torch.float32)

        return img_array, (vector, label)


    def __getname__(self, label: int) -> str:
        """
        Returns a descriptive name for a given label.

        @param label: label you want the name for.
        """
        return self.string_labels[label]


if __name__ == "__main__":
    """
    Test script that is only executed when this script is executed. Not during imports.
    """

    folderpath = "../../data/GTSRB/Final_Training/Images/"

    filepath = "../../data/concepts_per_class.csv"

    dataset = ImageDataset(folderpath, filepath, 128, 128)

    img, (vec, lbl) = dataset[0]
    print(img.shape)
    print(img.min(), img.max())
    print(vec.shape)
    print(lbl)

    sample = dataset.__getitem__(65)
    print(sample[0])


