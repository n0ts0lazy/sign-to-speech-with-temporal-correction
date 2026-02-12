"""
model_definitions.py — Contains model architectures for ASL Sign Language project.
Includes:
  - ResNet18 (pretrained)
"""

from torch import nn
from torchvision import models

# ResNet18 model (transfer learning)

def make_resnet18(num_classes, pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

