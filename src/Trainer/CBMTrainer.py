import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import sys
from torch.utils.data import DataLoader

sys.path.append("..")

from Data.Data import ImageDataset
from Data.DataLoader import ImageDataLoader
from Models.CBMModel import CBMModel
from Models.Model import Model_CNN
from Models.Model2 import Model
from Models.SimpleModel1 import SimpleModel1

"""
Class that manages the training and validation of the CBM Model
"""
class CBMTrainer:
    """
    Initiates an instance of the trainer and defines loss functions and optimizers.

    @param device: cuda device for faster calculation
    @param model : cbm model we want to train
    @param train_loader: dataloader for training
    @param val_loader: dataloader for validation
    """
    def __init__(self, device: torch.device, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.device = device
        self.model = model


        self.train_loader = train_loader
        self.val_loader = val_loader

        # loss functions
        self.loss_fun = [nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()]

        # optimizer
        self.optimizer = [
            optim.Adam(self.model.cnn.parameters(), lr=0.0005),
            optim.Adam(self.model.concept.parameters(), lr=0.0005)
        ]

    """
    Freezes parameters of a given submodel.

    @param model: model you want to freeze.
    """
    def freeze(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False
    

    """
    Unfreezes parameters of a given submodel.

    @param model: model you want to unfreeze.
    """
    def unfreeze(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True 

    
    """
    Main function to start the training process. Responsible for initiating concept und label training.
    """
    def main(self):
        start = time.time()
        self.concept_training(8)
        self.label_training(8)
        end = time.time()
        total_time = end-start
        print(total_time)

    
    """
    First phase of training, where we train for best fitting concept vectors.

    @param num_epochs: Number of epochs the training last in worst case.
    """
    def concept_training(self, num_epochs: int) -> None:
        # Freezes second model
        self.freeze(self.model.concept)

        for epoch in range(num_epochs):
            print(f"\n Epoch {epoch + 1}/{num_epochs}")

            self.training(0)
            self.validation(0)

            # Calculate epoch statistics TODO

        # Freezes first model
        self.unfreeze(self.model.concept)
    

    """
    Second phase of training, where we train for best fitting labels.

    @param num_epochs: Number of epochs the training last in worst case.
    """
    def label_training(self, num_epochs: int) -> None:
        # Freezes first model
        self.freeze(self.model.cnn)
        
        for epoch in range(num_epochs):
            print(f"\n Epoch {epoch + 1}/{num_epochs}")

            self.training(1)
            self.validation(1)

            # Calculate epoch statistics TODO

        # Freezes second model
        self.unfreeze(self.model.cnn)

    
    """
    Training function, that works for both concept and label training.

    @param phase: concept training needs phase 0 and label training needs phase 1
    """
    def training(self, phase: int) -> None:
        print("Training")
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_id, (data, (vector, label)) in enumerate(self.train_loader):
            data = data.to(self.device)

            target = vector if (phase == 0) else label
            target = target.to(self.device)

            self.optimizer[phase].zero_grad()
            
            c_logits, _, l_logits, _ = self.model(data)
            output = c_logits if (phase == 0) else l_logits

            loss = self.loss_fun[phase](output, target)
            loss.backward()
            self.optimizer[phase].step()

            total_loss += loss.item()

            if (phase == 0):
                predicted = (torch.sigmoid(output) > 0.5).float()
                total_samples += target.numel()
            else:
                predicted = torch.argmax(output, dim=1)
                total_samples += target.size(0)
            
            correct_predictions += (predicted == target).sum().item()

            # Print progress
            if batch_id % 100 == 0:
                current_loss = total_loss / (batch_id + 1)
                train_acc = 100. * correct_predictions / total_samples
                print(f"Training Batch {batch_id:3d}: Loss = {current_loss:.4f} | Acc: {train_acc:.2f}%")
        return
            

    """
    Validation function, that works for both concept and label validation.
    
    @param phase: concept validation needs phase 0 and label validation needs phase 1
    """
    def validation(self, phase: int) -> None:
        print("VALIDATE")
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch_id, (data, (vector, label)) in enumerate(self.val_loader):
                data = data.to(self.device)
                target = vector if (phase == 0) else label
                target = target.to(self.device)

                c_logits, _, l_logits, _ = self.model(data)
                output = c_logits if (phase == 0) else l_logits

                loss = self.loss_fun[phase](output, target)
                total_loss += loss.item()

                if (phase == 0):
                    predicted = (torch.sigmoid(output) > 0.5).float()
                    total_samples += target.numel()
                else:
                    predicted = torch.argmax(output, dim=1)
                    total_samples += target.size(0)
                
                correct_predictions += (predicted == target).sum().item()
               
                if batch_id % 100 == 0:
                    current_loss = total_loss / (batch_id + 1)
                    train_acc = 100. * correct_predictions / total_samples
                    print(f"Training Batch {batch_id:3d}: Loss = {current_loss:.4f} | Acc: {train_acc:.2f}%")
        return


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    image_path = "../../data/GTSRB/Final_Training/Images/"
    csv_path = "../../data/concepts_per_class.csv"
    destination_path = "../../models/cnn/model1.pth"
    loader = ImageDataLoader(image_path=image_path,
                             csv_path=csv_path,
                             pixelsx=128, pixelsy=128,
                             batch_size=32,
                             train_portion=0.8,
                             is_own_model=True)
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()
    cnn_model = SimpleModel1(43).to(device)
    concept_model = Model(43, 43).to(device)
    model = CBMModel(cnn_model, concept_model).to(device)
    trainer = CBMTrainer(device, model, train_loader, val_loader)
    trainer.main()
    #torch.save(trained_model.state_dict(), destination_path)
    