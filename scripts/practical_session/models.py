"""
PyTorch Models for MNIST Classification

This module contains different neural network architectures for MNIST classification,
designed to demonstrate underfitting, good fitting, and overfitting behaviors.
"""
import torch.nn as nn
import torch.nn.functional as F


class TooSimpleNet(nn.Module):
    
    def __init__(self, input_size=784, num_classes=10):
        super(TooSimpleNet, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)


class GoodNet(nn.Module):

    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, dropout_prob=0.3):
        super(GoodNet, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class TooComplexNet(nn.Module):
    """
    Extremely complex model - almost guaranteed to overfit on MNIST
    This model has way too many parameters for the MNIST task
    """
    def __init__(self, input_size=784, num_classes=10):
        super(TooComplexNet, self).__init__()
        
        # Massive hidden layers - way too big for MNIST
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 128)
        self.fc10 = nn.Linear(128, 64)
        self.fc11 = nn.Linear(64, num_classes)
        
        # Absolutely NO dropout or regularization to encourage maximum overfitting
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        return self.fc11(x)


class ConvNet(nn.Module):
    
    def __init__(self, num_classes=10, dropout_prob=0.5):
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        # After 3 pooling operations: 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
