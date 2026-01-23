
import torch
from torch.utils.data import DataLoader, random_split
from Data.Data import ImageDataset
from Data.Data2 import ImageDataset2
from Seeding import seed_worker

class ImageDataLoader:
    """
    Initiates an instance of a Dataloader.

    @param image_path: Gives location of image data.
    @param csv_path: Gives location of csv file, which contains concept vectors.
    @param pixelsx: gives number of pixels in x dimension.
    @param pixelsy: gives numbre of pixels in y dimension.
    @param batch_size: gives batch size.
    @param train_portion: gives distribution of data into training and validation data.
    @param is_own_model: tells which dataset to use (dataset for own model or for pre trained model)
    """
    def __init__(self, image_path: str, csv_path: str, pixelsx: int, pixelsy: int, batch_size: int, train_portion: float, is_own_model: bool):

        #seed for reproducability
        self.seed=42

        # create Dataset
        if (is_own_model):
            self.dataset = ImageDataset2(image_path, csv_path, pixelsx, pixelsy)
        else:
            self.dataset = ImageDataset(image_path, csv_path, pixelsx, pixelsy)
        self.batch_size = batch_size
        self.train_portion = train_portion
        
        # Split dataset
        train_size = int(self.dataset.__len__() * self.train_portion)
        val_size = self.dataset.__len__() - train_size

        # fix seed for splitting
        split_generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size], generator=split_generator)

    def get_train_loader(self):
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.seed)
        train_loader = DataLoader(self.train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True, 
                                  num_workers=4, 
                                  worker_init_fn=seed_worker,
                                  generator=loader_generator)
        return train_loader

    
    def get_val_loader(self):
        val_loader = DataLoader(self.val_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=False, 
                                  num_workers=4,
                                  worker_init_fn=seed_worker)
        return val_loader




        