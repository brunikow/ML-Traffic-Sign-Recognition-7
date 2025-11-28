import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class Model_CNN(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Model_CNN, self).__init__()
        self.num_classes = num_classes

        # pretrained network
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        
        # output of the last layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = torch.nn.Identity()

        # replaces last layer
        self.classifier = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        probability = torch.sigmoid(logits)
        return probability


