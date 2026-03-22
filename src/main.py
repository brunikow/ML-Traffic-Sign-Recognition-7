import torch
import sys
import requests
import zipfile
import os
import time

from pathlib import Path

from torch.utils.data import DataLoader
from Seeding import set_seed
from Data.Data import ImageDataset
from Data.DataLoader import ImageDataLoader
from Models.Model import Model_CNN
from Models.Model2 import Model
from Models.SimpleModel1 import SimpleModel1
from Models.CBMModel import CBMModel
from Trainer.CBMTrainer import CBMTrainer
from CLI.evaluate import eval

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

"""
This class is the entry point of our program. It gives an easier access for the user to tune parameters.
"""

class Main:
    """
    Initiates a main object. takes all needed information and saves them.
    """
    
    def __init__(self, device: str, is_own_model: bool, batch_size: int, learning_rate: float, train_portion: int, c_epochs: int, l_epochs: int, patience: int):
        self.image_path = DATA_DIR / "GTSRB/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
        self.csv_path = DATA_DIR / "concepts_per_class.csv"
        self.device = device
        self.is_own_model = is_own_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_portion = train_portion
        self.c_epochs = c_epochs
        self.l_epochs = l_epochs
        self.patience = patience

        self.model = CBMModel


    """
    Entry point of the Main class. Initiates all needed objects like device, loader, models and trainer. Than it starts the training process.
    """
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

    """
    Safes the model at a given destination.

    @param destination_path: Location of the model file.
    """
    def save_model(self, destination_path):
        torch.save(self.model.state_dict(), destination_path)
        return


    """
    Safes meta data like learning rate and batch_size for a trained model.

    @param destination_path: Location of the meta data file.
    """
    def save_meta_data(self, destination_path):
        with open(destination_path, "w") as file:
            file.write("# Metadata for " + str(destination_path) + "\n\n")

            file.write("- device: " + self.device + "\n")
            file.write("- model_variant: " + ("self written model" if (self.is_own_model) else "efficientNet model") + "\n")
            file.write("- batch_size: " + str(self.batch_size) + "\n")
            file.write("- learning_rate: " + str(self.learning_rate) + "\n")
            file.write("- train_portion: " + str(self.train_portion) + "\n")
            file.write("- c_epochs: " + str(self.c_epochs) + "\n")
            file.write("- l_epochs: " + str(self.l_epochs) + "\n")
            file.write("- patience: " + str(self.patience) + "\n")

        return

def init():
    timer_start = time.time()
    init_download("GTSRB_Final_Training_Images")
    init_download("GTSRB_Final_Test_Images")
    init_download("GTSRB_Final_Test_GT")
    timer_end = time.time()

    subdirs = ["cnnmodel", "cbmmodel", "conceptmodel"]
    for subdir in subdirs:
        os.makedirs(MODEL_DIR / subdir, exist_ok=True)

    minutes, seconds = divmod(timer_end - timer_start, 60)

    print(f"Initializing the project took {int(minutes)}m {seconds:.2f}s")

def init_download(file):
    print(f"Downloading {file} from the GTSRB")
    url = f"https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/{file}.zip"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(DATA_DIR / file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        os.makedirs(DATA_DIR / "GTSRB" / file, exist_ok=True)
        with zipfile.ZipFile(DATA_DIR / file, 'r') as f:
            f.extractall(DATA_DIR / "GTSRB" / file)

        os.remove(DATA_DIR / file)

if __name__ == "__main__":
    # set your configuration here!!!
    set_seed(42)

    print(BASE_DIR)
    if len(sys.argv) > 1:
        if sys.argv[1] == "init":
            init()

        elif sys.argv[1] == "run":
            own_model = False
            if len(sys.argv) > 2:
                if sys.argv[2] == "own_model":
                    print("Training is set without the EfficientNetV2 backbone")
                    own_model = True

            starter = Main(device = "cuda:1", 
                        is_own_model = own_model, 
                        batch_size = 32, 
                        learning_rate = 0.003, 
                        train_portion = 0.8, 
                        c_epochs = 40 if own_model else 20,
                        l_epochs = 20, 
                        patience = 5 if own_model else 3
                        )
            starter.main()
            destination = MODEL_DIR / "cbmmodel/final_model" if own_model else "cbmmodel/final_model_ef"
            starter.save_model(destination.with_suffix(".pth"))
            starter.save_meta_data(destination.with_suffix(".md"))

        elif sys.argv[1] == "eval":
            eval()