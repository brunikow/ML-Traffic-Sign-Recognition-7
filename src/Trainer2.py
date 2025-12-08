import torch
import torch.nn as nn
import torch.optim as optim
from Data import ImageDataset
from Model import Model_CNN
from DataLoader import ImageDataLoader
from Model2 import Model
import copy
import time

class Trainer2:
    def __init__(self, device, model, cnn_model_path, train_loader, val_loader):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0005)

        # load cnn_model
        cnn_model = Model_CNN(43)
        cnn_model.load_state_dict(torch.load(cnn_model_path))
        cnn_model.eval()
        cnn_model.to(device)

        self.cnn_model = cnn_model

    def simple_training(self, num_epochs, patience):
        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range (num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Trainig phase
            self.model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            print("TRAINING")
            for batch_idx, (data, (_, label)) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                with torch.no_grad():
                    c_vectors = self.cnn_model(data)
                    # c_vectors = torch.sigmoid(c_vectors)
                    c_vectors = c_vectors.to(self.device)


                self.optimizer.zero_grad()

                outputs = self.model(c_vectors)

                loss = self.criterion(outputs, label)

                loss.backward()
                
                self.optimizer.step()

                total_loss += loss.item()

                
                predicted = torch.argmax(outputs, dim=1)
                correct_predictions += (predicted == label).sum().item()
                total_samples += label.size(0) 

                # Print progress
                if batch_idx % 100 == 0:
                    current_loss = total_loss / (batch_idx + 1)
                    train_acc = 100. * correct_predictions / total_samples
                    print(f"Training Batch {batch_idx:3d}: Loss = {current_loss:.4f} | Acc: {train_acc:.2f}%")

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            print("VALIDATION")
            with torch.no_grad():
                for batch_idx, (data, (_, label)) in enumerate(self.val_loader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    
                    c_vectors = self.cnn_model(data)
                    #c_vectors = torch.sigmoid(c_vectors)
                    c_vectors = c_vectors.to(self.device)

                    output = self.model(c_vectors)
                    loss = self.criterion(output, label)
                    val_loss += loss.item()

                    predicted = (torch.argmax(output, dim=1))
                    val_correct += (predicted == label).sum().item()
                    val_total += label.size(0)

                    # Print progress
                    if batch_idx % 100 == 0:
                        current_loss = val_loss / (batch_idx + 1)
                        val_acc = 100. * val_correct / val_total
                        print(f"Validation Batch {batch_idx:3d}: Loss = {current_loss:.4f} | Acc: {val_acc:.2f}%")



            # Calculate epoch statistics
            val_epoch_loss = val_loss / len(self.val_loader)
            val_epoch_acc = 100. * val_correct / val_total
            print(f"  Epoch {epoch + 1} Summary: Loss = {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.2f}%")
            
            """
            Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("stopped early didnt improve in a bit")
                    break
            """

        self.model.load_state_dict(best_model_wts)
        stop = time.time()
        total_time = stop - start
        print(f"\nTraining completed! time passed: {total_time}")

        return self.model

if __name__ == "__main__":
    # set up device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    image_path = "../data/GTSRB/Final_Training/Images/"
    csv_path = "../data/concepts_per_class.csv"
    destination_path = "../models/cnn/model1_8e.pth"
    loader = ImageDataLoader(image_path=image_path,
                             csv_path=csv_path,
                             pixelsx=128, pixelsy=128,
                             batch_size=32,
                             train_portion=0.8)
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()
    model = Model(43, 43)
    model.to(device)
    trainer = Trainer2(device, model, destination_path, train_loader, val_loader)
    trained_model = trainer.simple_training(8, 3)


