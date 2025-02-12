import torch
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Grayscale (1 channel)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

