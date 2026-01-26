import sys
sys.path.append("..")  # to allow imports from overarching directory
import torch
from torch.utils.data import DataLoader
from TestData import TestDataset
from Seeding import seed_worker

class TestDataLoader:
    """
    Initiates an instance of a Dataloader.

    @param image_path: Gives location of image data.
    @param concept_csv: Gives location of csv file, which contains concept vectors.
    @param label_csv: Gives location of csv file, which contains labels.
    @param pixelsx: gives number of pixels in x dimension.
    @param pixelsy: gives numbre of pixels in y dimension.
    @param batch_size: gives batch size.
    @param is_own_model: tells which dataset to use (dataset for own model or for pre trained model)
    """
    def __init__(self, image_path: str, concept_csv: str, label_csv: str, pixelsx: int, pixelsy: int, batch_size: int, is_own_model: bool):

        #seed for reproducability
        self.seed=42

        # create Dataset
        self.dataset = TestDataset(image_path, concept_csv, label_csv, pixelsx, pixelsy, is_own_model)
        self.batch_size = batch_size
       
    def get_test_loader(self):   
        test_loader = DataLoader(self.dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=False,                # no shuffling for test data necessary 
                                  num_workers=4, 
                                  worker_init_fn=seed_worker)
        return test_loader

if __name__ == "__main__":
    """
    Test script that is only executed when this script is executed. Not during imports.
    """

    folderpath = "../../data/GTSRB/Final_Test/Images/"
    label_csv = "../../data/GTSRB/Final_Test_GT/GT-final_test.csv"
    concept_csv = "../../data/concepts_per_class.csv"

    test_loader = TestDataLoader(folderpath, concept_csv, label_csv, 128, 128, 32, False)
    loader = test_loader.get_test_loader()
    sample = next(iter(loader))
    images, (vectors, labels) = sample
    print(images.shape)
    print(vectors.shape)
    print(labels.shape) 
    print(images[0])
    print(vectors[0])
    print(labels[0])    
    print(len(images))
    

        