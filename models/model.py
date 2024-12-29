import torch
import torch.nn as nn
import torch.nn.functional as F

DROP_OUT = 0.1
class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()

        # Block 1 (Receptive Field: 3 -> 7 with dilation)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, dilation=1, padding=1, groups=1),  # RF: 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2, groups=32),  # RF: 7 (Depthwise)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Conv2d(32, 32, kernel_size=1),  # Pointwise convolution (RF unchanged: 7)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Block 2 (Receptive Field: 7 -> 15 -> 23)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, dilation=2, padding=2, groups=32),  # RF: 15 (Depthwise)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Conv2d(64, 64, kernel_size=3, dilation=3, padding=3, groups=64),  # RF: 23 (Depthwise)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Conv2d(64, 64, kernel_size=1),  # Pointwise convolution (RF unchanged: 23)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Block 3 (Receptive Field: 23 -> 39 -> 55)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=3, padding=3, groups=64),  # RF: 39 (Depthwise)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Conv2d(128, 128, kernel_size=3, dilation=4, padding=4, groups=128),  # RF: 55 (Depthwise)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Conv2d(128, 128, kernel_size=1),  # Pointwise convolution (RF unchanged: 55)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Block 4 (Receptive Field: 55 -> 87 -> 119)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, dilation=4, padding=4, groups=128),  # RF: 87 (Depthwise)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Conv2d(256, 256, kernel_size=3, dilation=5, padding=5, groups=256),  # RF: 119 (Depthwise)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Conv2d(256, 256, kernel_size=1),  # Pointwise convolution (RF unchanged: 119)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global pooling (RF covers entire input)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(DROP_OUT)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x