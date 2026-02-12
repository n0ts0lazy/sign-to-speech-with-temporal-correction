import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s

def create_mvit_model(num_classes=2000, pretrained=True):
    model = mvit_v2_s(weights="DEFAULT" if pretrained else None)
    in_features = model.head[1].in_features
    model.head[1] = nn.Linear(in_features, num_classes)
    return model
