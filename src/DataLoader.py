from torch.utils.data import DataLoader, random_split
from Data import ImageDataset

class ImageDataLoader:
    def __init__(self, image_path, csv_path, pixelsx, pixelsy, batch_size, train_portion):

        # create Dataset
        self.dataset = ImageDataset(image_path, csv_path, pixelsx, pixelsy)
        self.batch_size = batch_size
        self.train_portion = train_portion
        
        # Split dataset
        train_size = int(self.dataset.__len__() * self.train_portion)
        val_size = self.dataset.__len__() - train_size

        # use of random maybe, need to use a seed for reproducability
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def get_train_loader(self):
        train_loader = DataLoader(self.train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True, 
                                  num_workers=4)
        return train_loader

    
    def get_val_loader(self):
        val_loader = DataLoader(self.val_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=False, 
                                  num_workers=4)
        return val_loader




        