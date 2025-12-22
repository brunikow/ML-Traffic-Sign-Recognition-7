import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import sys

sys.path.append("..")

from Data.Data import ImageDataset
from Data.DataLoader import ImageDataLoader
from Models.CBMModel import CBMModel
from Models.Model import Model_CNN
from Models.Model2 import Model
from sklearn.metrics import precision_score, recall_score, f1_score

class CBMTrainer:
    def __init__(self, device, model, train_loader, val_loader, patience = 5):
        self.device = device
        self.model = model
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        # loss functions
        self.loss_fun = [nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()]

        self.optimizer = [
            optim.Adam(self.model.cnn.parameters(), lr=0.0005),
            optim.Adam(self.model.concept.parameters(), lr=0.0005)
        ]

        #early stopping and history metrics

        self.patience = patience

        self.best_value_loss = float('inf')
        self.current_patience = 0

        self.history = {
            "train_loss" : [],
            "per-concept train_acc" : [],
            "val_loss" : [],
            "per-concept val_acc" : []}


    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
    

    def unfreeze(self, model):
        for param in model.parameters():
            param.requires_grad = True 

    
    def main(self):
        start = time.time()
        self.concept_training(10)
        self.label_training(8)
        end = time.time()
        total_time = end-start
        print(total_time)

    
    def concept_training(self, num_epochs):
        self.freeze(self.model.concept)

        for epoch in range(num_epochs):
            print(f"\n Epoch {epoch + 1}/{num_epochs}")

            self.training(0)
            self.validation(0)
            ##IMPORTANT EARLY STOPPING FUNCTION
            # Calculate epoch statistics TODO

        self.unfreeze(self.model.concept)
    

    def label_training(self, num_epochs):
        self.freeze(self.model.cnn)
        
        for epoch in range(num_epochs):
            print(f"\n Epoch {epoch + 1}/{num_epochs}")

            self.training(1)
            self.validation(1)

            # Calculate epoch statistics TODO

        self.unfreeze(self.model.cnn)

    
    def training(self, phase):
        print("Training")
        self.model.train()

        total_loss = 0.0
        total_samples, vector_correct, concept_correct, concept_total, class_correct = 0,0,0,0,0

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

                #per vector accuracy
                #a vector only counts as correct if all items in it are correct (strict implementation of correctness)
                batch_size = target.size(0)
                total_samples += batch_size
                vector_correct += (predicted == target).all(dim=1).sum().item()
                
                #per concept accuracy
                concept_correct += (predicted == target).sum().item()
                concept_total += target.numel()

            else:
                predicted = torch.argmax(output, dim=1)
                total_samples += target.size(0)
                class_correct += (predicted == target).sum().item()

            # Print progress
            if batch_id % 100 == 0:
                current_loss = total_loss / (batch_id + 1)
                if phase == 0:
                    vector_acc = 100. * vector_correct / total_samples if total_samples > 0 else 0
                    concept_acc = 100. * concept_correct / concept_total if concept_total > 0 else 0
                    print(f"Training Batch {batch_id:3d}: Loss = {current_loss:.4f} | Vector Acc: {vector_acc:.4f}% | Per-Concept Acc: {concept_acc:.4f}%")
                else:
                    train_acc = 100. * class_correct / total_samples
                    print(f"Training Batch {batch_id:3d}: Loss = {current_loss:.4f} | Acc: {train_acc:.4f}%")
        return
    
    def validation(self, phase):
        print("VALIDATE")
        self.model.eval()

        total_loss = 0.0
        total_samples, vector_correct, concept_correct, concept_total, class_correct = 0,0,0,0,0

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

                    #per vector accuracy
                    #a vector only counts as correct if all items in it are correct (strict implementation of correctness)
                    batch_size = target.size(0)
                    total_samples += batch_size
                    vector_correct += (predicted == target).all(dim=1).sum().item()
                
                    #per concept accuracy
                    concept_correct += (predicted == target).sum().item()
                    concept_total += target.numel()

                else:
                    predicted = torch.argmax(output, dim=1)
                    total_samples += target.size(0)
                    class_correct += (predicted == target).sum().item()

                average_loss = total_loss / len(self.val_loader)

                # Print progress
                if batch_id % 100 == 0:
                    current_loss = total_loss / (batch_id + 1)
                    if phase == 0:
                        vector_acc = 100. * vector_correct / total_samples if total_samples > 0 else 0
                        concept_acc = 100. * concept_correct / concept_total if concept_total > 0 else 0
                        print(f"Validation Batch {batch_id:3d}: Loss = {current_loss:.4f} | Average Loss = {average_loss:.4f} | Vector Acc: {vector_acc:.4f}% | Per-Concept Acc: {concept_acc:.4f}%")
                    else:
                        train_acc = 100. * class_correct / total_samples
                        print(f"Validation Batch {batch_id:3d}: Loss = {current_loss:.4f} | Acc: {train_acc:.4f}%")
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
                             train_portion=0.8)
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()
    cnn_model = Model_CNN(43).to(device)
    concept_model = Model(43, 43).to(device)
    model = CBMModel(cnn_model, concept_model).to(device)
    trainer = CBMTrainer(device, model, train_loader, val_loader)
    trainer.main()
    #torch.save(trained_model.state_dict(), destination_path)
    