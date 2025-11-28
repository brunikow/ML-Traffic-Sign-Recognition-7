import torch
import torch.nn as nn
import torch.optim as optim
from Data import ImageDataset
from Model import Model_CNN
from DataLoader import ImageDataLoader


class Trainer:
    def __init__(self, device, model, train_loader, val_loader):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr = 0.001)

    def simple_training(self, num_epochs):
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                model.train()
                total_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                for batch_idx, (data, (vector, label)) in enumerate(train_loader):
                    # Move data to device
                    data = data.to(device)
                    vector = vector.to(device)

                    # Zero the gradients (important!)
                    self.optimizer.zero_grad()
                    
                    # Forward pass: compute predictions
                    outputs = model(data)
                    
                    # Compute loss
                    loss = self.criterion(outputs, vector)
                    
                    # Backward pass: compute gradients
                    loss.backward()
                    
                    # Update parameter, what we have done manually before
                    self.optimizer.step()
                    
                    # Track statistics
                    total_loss += loss.item()
                    
                    # Print progress
                    current_loss = total_loss / (batch_idx + 1)
                    print(f"  Batch {batch_idx:3d}: Loss = {current_loss:.4f}")
                
                # Calculate epoch statistics
                epoch_loss = total_loss / len(train_loader)
                print(f"  Epoch {epoch + 1} Summary: Loss = {epoch_loss:.4f}")
            
                print(f"\nTraining completed!")
                return model       

if __name__ == "__main__":
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "../data/GTSRB/Final_Training/Images/"
    csv_path = "../data/concepts_per_class.csv"
    loader = ImageDataLoader(image_path=image_path,
                             csv_path=csv_path,
                             pixelsx=128, pixelsy=128,
                             batch_size=128,
                             train_portion=0.8)

    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()
    model = Model_CNN(43)
    trainer = Trainer(device, model, train_loader, val_loader)
    trainer.simple_training(8)

