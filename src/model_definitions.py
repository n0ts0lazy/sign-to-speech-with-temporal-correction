"""
model_definitions.py — Contains model architectures for ASL Sign Language project.
Includes:
  - ResNet18 (pretrained)
  - MobileNetV3 (pretrained)
  - CNN + LSTM hybrid (for future temporal modeling)
"""

from torch import nn
from torchvision import models

# ResNet18 model (transfer learning)

def make_resnet18(num_classes, pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# MobileNetV3 model (lightweight)

def make_mobilenet(num_classes, pretrained=True):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# CNN + LSTM hybrid (temporal extension)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1):
        super(CNNLSTM, self).__init__()

        # Simple CNN backbone for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(input_size=64 * 56 * 56, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Final classifier
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # CNN expects (B, C, H, W), but we can simulate sequences in future (for video)
        b, c, h, w = x.size()
        features = self.cnn(x)
        features = features.view(b, 1, -1)  # add temporal dimension (seq_len = 1)
        _, (h_n, _) = self.lstm(features)
        out = self.fc(h_n[-1])
        return out


def make_cnn_lstm(num_classes):
    return CNNLSTM(num_classes)
