import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Model that takes concept vectors as input and gives back a label.
"""
class SimpleModel1(torch.nn.Module):
    """
    Initiates an instance of the model.
    @param num_concepts: takes length of concept vector
    @param num_labels: takes length of the output
    """
    def __init__(self,num_classes: int) -> None:
        super(SimpleModel1, self).__init__()
        self.num_classes = num_classes

        # Convolutional layers and normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer (for pixelx and pixelsy = 128)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # against overfitting
        self.dropout = nn.Dropout(0.4)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
       
        return x


