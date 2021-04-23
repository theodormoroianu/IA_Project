import torch as th
import torch.nn as nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18()
        self.fc = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

class ConvUnit(nn.Module):
    def __init__(self, in_f, out_f, ker=3, max_pull=False):
        super().__init__()
        if max_pull == False:
            self.net = nn.Sequential (
                nn.Conv2d(in_f, out_f, kernel_size=ker, padding=ker//2),
                nn.BatchNorm2d(out_f),
                nn.ReLU()
            )
        else:
            self.net = nn.Sequential (
                nn.Conv2d(in_f, out_f, kernel_size=ker, padding=ker//2),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
    def forward(self, x):
        return self.net(x)

class FcUnit(nn.Module):
    def __init__(self, in_f, out_f, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential (
            nn.Linear(in_f, out_f),
            nn.BatchNorm1d(out_f),
            nn.Dropout(dropout),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)



class Clasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvUnit(3, 32, ker=7, max_pull=True),
            ConvUnit(32, 64, ker=5, max_pull=True),
            ConvUnit(64, 128, ker=3, max_pull=True),

            nn.Flatten(),

            FcUnit(128 * 6 * 6, 1000),
            FcUnit(1000, 500),
            FcUnit(500, 3)
        )

    def forward(self, x):
        return self.net(x)