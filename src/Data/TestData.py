import numpy as np
from PIL import Image
import glob
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.models import EfficientNet_V2_S_Weights
# Fuer README addition spaeter, fuer testdataset wird noch GTSRB_Final_Test_GT.zip benoetigt

class TestDataset(Dataset):
    """
    A class used to create and manage a dataset out of given data.
    """
    def __init__(self, image_path: str, concept_csv: str, label_csv: str, pixelsx: int, pixelsy: int, is_own_model: bool) -> None:
        """
        Initializes a dataset out of given data by.

        1. reading a csv file with labels and vectors.
        2. going through the data, which is structured in a folder structure.

        @param image_path: path to the directory which contains all the images.
        @param label_csv: path to the csv-file that contains the labels.
        @param concept_csv: path to the csv-file that contains the concepts.
        @param pixelsx: contains the uniform pixel number all images will get in x axis.
        @param pixelsy: contains the uniform pixel number all images will get in y axis.
        @param is_own_model: tells which model type should be handled
        """
        self.image_folderpath = image_path
        self.label_csv = label_csv
        self.concept_csv = concept_csv
        self.pixelsx = pixelsx
        self.pixelsy = pixelsy
        self.is_own_model = is_own_model

        #self.samples = []
        self.sample_paths = []
        self.labels = []
        self.class_names = []
        self.concept_vectors = []

        # Read csv file and extract class names and vectors
        df = pd.read_csv(concept_csv, skiprows=0)
        self.class_names = df.iloc[:, 1].to_numpy()
        self.concept_vectors = df.iloc[:, 2:].to_numpy()
        
        # Read label_csv file and extract filenames and ClassId labels
        labels_df = pd.read_csv(label_csv, delimiter=';')
        self.image_files = labels_df['Filename'].tolist()
        self.labels = labels_df['ClassId'].tolist()
        
        # preparation for efficientNet
        if not self.is_own_model:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.transform = weights.transforms()
        else: 
            self.transform = None
        
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.image_files)

    def get_img_name__(self, idx:int) -> str:
        """Returns a filename for a given index."""
        return self.image_files[idx]

    def get_label__(self, idx:int) -> int:
        """Returns a label for a given index."""
        return self.labels[idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns image and label for a given index.
        
        @param idx: index of the sample
        @return: Tuple of (image_tensor, label_tensor)
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folderpath, img_name)
        img = Image.open(img_path).convert("RGB")
        
        if self.is_own_model:
            img = img.resize((self.pixelsx, self.pixelsy))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.transpose(img_array, (2, 0, 1))
            img_tensor = torch.tensor(img_array) / 255.0
        else:
            img_tensor = self.transform(img)

        # sample = self.samples[idx]
        label =  self.labels[idx]
        concept_vector = self.concept_vectors[label]

        # Transform everything to tensors (needed for ML usage)
        # sample = torch.tensor(sample, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        concept_tensor = torch.tensor(concept_vector, dtype=torch.float32)
        
        return img_tensor, (concept_tensor, label_tensor)


if __name__ == "__main__":
    """
    Test script that is only executed when this script is executed. Not during imports.
    """

    folderpath = "../../data/GTSRB/Final_Test/Images/"
    label_csv = "../../data/GTSRB/Final_Test_GT/GT-final_test.csv"
    concept_csv = "../../data/concepts_per_class.csv"

    test_dataset = TestDataset(folderpath, concept_csv, label_csv, 128, 128, False)

    sample = test_dataset.__getitem__(65)
    print(sample[0])
    