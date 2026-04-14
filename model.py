import torch
import torch.nn as nn
from torchvision import models

class DeepfakeModel(nn.Module):
    def __init__(self):
        super(DeepfakeModel, self).__init__()
        
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        
        # Replace final layer
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Real or Fake
        )

    def forward(self, x):
        return self.model(x)
