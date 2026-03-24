import torch
import torch.nn as nn
import torch.nn.functional as F

class CBMModel(nn.Module):
    """

    """

    def __init__(self, cnn_model: nn.Module, concept_model:nn.Module):
        super().__init__()
        self.cnn = cnn_model
        self.concept = concept_model


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_logits = self.cnn(x)

        c_propabilities = torch.sigmoid(c_logits)

        l_logits = self.concept(c_propabilities)

        l_propabilities = torch.sigmoid(l_logits)

        return c_logits, c_propabilities, l_logits, l_propabilities 

