import torch
from torch.utils.data import DataLoader

from Data.Data import ImageDataset
from Data.DataLoader import ImageDataLoader
from Models.Model import Model_CNN
from Models.Model2 import Model
from Models.SimpleModel1 import SimpleModel1
from Models.CBMModel import CBMModel
from Trainer.CBMTrainer import CBMTrainer

class Main:
    def __init__(self, device: str, is_own_model: bool, batch_size: int, learning_rate: float, train_portion: int, c_epochs: int, l_epochs: int, patience: int):
        self.image_path = "../data/GTSRB/Final_Training/Images/"
        self.csv_path = "../data/concepts_per_class.csv"
        self.device = device
        self.is_own_model = is_own_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_portion = train_portion
        self.c_epochs = c_epochs
        self.l_epochs = l_epochs
        self.patience = patience

        self.model = CBMModel

    def main(self):
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        loader = ImageDataLoader(image_path=self.image_path,
                                 csv_path=self.csv_path,
                                 pixelsx=128,
                                 pixelsy=128,
                                 batch_size=self.batch_size,
                                 train_portion=self.train_portion,
                                 is_own_model=self.is_own_model)
        train_loader = loader.get_train_loader()
        val_loader = loader.get_val_loader()

        if (self.is_own_model):
            cnn_model = SimpleModel1(43).to(device)
        else:
            cnn_model = Model_CNN(43).to(device)

        concept_model = Model(43, 43).to(device)
        cbm_model = CBMModel(cnn_model, concept_model)

        trainer = CBMTrainer(device, cbm_model, train_loader, val_loader, self.patience, self.learning_rate, self.c_epochs, self.l_epochs)

        self.model = trainer.main()

        return


    def safe_model(self, destination_path):
        torch.save(self.model.state_dict(), destination_path)
        return


    def safe_meta_data(self, destination_path):
        with open(destination_path, "w") as file:
            file.write("# Metadata for " + destination_path + "\n\n")

            file.write("- device: " + self.device + "\n")
            file.write("- model_variant: " + ("self written model" if (self.is_own_model) else "efficientNet model") + "\n")
            file.write("- batch_size: " + str(self.batch_size) + "\n")
            file.write("- learning_rate: " + str(self.learning_rate) + "\n")
            file.write("- train_portion: " + str(self.train_portion) + "\n")
            file.write("- c_epochs: " + str(self.c_epochs) + "\n")
            file.write("- l_epochs: " + str(self.l_epochs) + "\n")
            file.write("- patience: " + str(self.patience) + "\n")

        return


if __name__ == "__main__":
    # set your configuration here!!!
    starter = Main(device = "cuda:1", 
                   is_own_model = True, 
                   batch_size = 32, 
                   learning_rate = 0.005, 
                   train_portion = 0.8, 
                   c_epochs = 1, 
                   l_epochs = 1, 
                   patience = 4
                   )

    starter.main()

    destination = "../models/cnn/modelx"

    starter.safe_model(destination + ".pth")
    starter.safe_meta_data(destination + ".md")