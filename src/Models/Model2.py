import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class Model(torch.nn.Module):
    def __init__(self,num_concepts: int, num_labels: int) -> None:
        super(Model, self).__init__()
        self.fc1 = nn.Linear(num_concepts, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_labels)
        #self.fc4 = nn.Linear(500, 50)
        #self.fc5 = nn.Linear(50, num_labels)

        # against overfitting
        self.dropout = nn.Dropout(0.4)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        #x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc4(x))
        #x = self.dropout(x)
        #logits = self.fc5(x)
        
        return logits

