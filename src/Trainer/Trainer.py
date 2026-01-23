import torch
import torch.nn as nn
import torch.optim as optim
from Data.Data import ImageDataset
from Models.Model import Model_CNN
from Data.DataLoader import ImageDataLoader
import copy
import time
from Seeding import set_seed

class Trainer:
    def __init__(self, device, model, train_loader, val_loader):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)

    def simple_training(self, num_epochs, patience):
            start = time.time()

            best_model_wts = copy.deepcopy(self.model.state_dict())
            best_val_loss = float('inf')
            epochs_no_improve = 0

            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

                # Training
                total_loss, correct_predictions, total_samples = self.training()

                # Validation
                val_loss = self.validation()
                    
                # Calculate epoch statistics
                epoch_loss = total_loss / len(self.train_loader)
                train_acc = 100. * correct_predictions / total_samples
                print(f"  Epoch {epoch + 1} Summary: Loss = {epoch_loss:.4f} | Acc: {train_acc:.2f}%")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("stopped early didnt improve in a bit")
                        break
            
            self.model.load_state_dict(best_model_wts)

            stop = time.time()
            total_time = stop-start
            print(f"\nTraining completed! time passed: {total_time}")

            return self.model
    

    def training(self):
        # Training phase
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        print("TRAINING")
        for batch_idx, (data, (vector, label)) in enumerate(self.train_loader):
            # Move data to device
            data = data.to(self.device)
            vector = vector.to(self.device)

            # Zero the gradients (important!)
            self.optimizer.zero_grad()
            
            # Forward pass: compute predictions
            outputs = self.model(data)
            
            # Compute loss
            loss = self.criterion(outputs, vector)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update parameter, what we have done manually before
            self.optimizer.step()
            
            # Track statistics
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == vector).sum().item()
            total_samples += vector.numel() 

            # Print progress
            if batch_idx % 100 == 0:
                current_loss = total_loss / (batch_idx + 1)
                train_acc = 100. * correct_predictions / total_samples
                print(f"Training Batch {batch_idx:3d}: Loss = {current_loss:.4f} | Acc: {train_acc:.2f}%")

        return total_loss, correct_predictions, total_samples


    def validation(self):
        # Validation phase
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        print("VALIDATION")
        with torch.no_grad():
                for batch_idx, (data, (vector, label)) in enumerate(self.val_loader):
                    data = data.to(self.device)
                    vector = vector.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, vector)

                    val_loss += loss.item()
                    
                    predicted = (torch.sigmoid(output) > 0.5).float()
                    val_correct += (predicted == vector).sum().item()
                    val_total += vector.numel()

                    if batch_idx % 100 == 0:
                        current_loss = val_loss / (batch_idx + 1)
                        val_acc = 100. * val_correct / val_total
                        print(f"Validation: Batch {batch_idx:3d}: Loss = {current_loss:.4f} | Validation Acc: {val_acc:.2f}%")
        return val_loss





if __name__ == "__main__":
    # set seed for reproducability
    set_seed(42)
    # set up device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    image_path = "../data/GTSRB/Final_Training/Images/"
    csv_path = "../data/concepts_per_class.csv"
    destination_path = "../models/cnn/model1.pth"
    loader = ImageDataLoader(image_path=image_path,
                             csv_path=csv_path,
                             pixelsx=128, pixelsy=128,
                             batch_size=32,
                             train_portion=0.8,
                             is_own_model=False)
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()
    model = Model_CNN(43)
    trainer = Trainer(device, model, train_loader, val_loader)
    trained_model = trainer.simple_training(14, 5)
    torch.save(trained_model.state_dict(), destination_path)
    

"""
TODO

- per concept accuracy [1, 2, 3, 4, ... , 43]
- visualization

"""
