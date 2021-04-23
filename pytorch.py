# %%
# Include usual stuff.
import torch as th
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as td
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from os import mkdir
from shutil import copyfile, copytree
import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import sys
import torch as th
import submission
import torchvision.transforms.functional as T

best_act = 0.


# %%
TRANF_POWER = 0.1

train_transform = transforms.Compose([
        transforms.Resize(150),
        # transforms.Grayscale(),
        # transforms.ColorJitter(2 * TRANF_POWER, 2 * TRANF_POWER, TRANF_POWER, TRANF_POWER),
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=20),
        # transforms.RandomCrop(50, padding=1),
        # transforms.RandomAutocontrast(),
        # transforms.RandomAdjustSharpness(0.95),
        # transforms.RandomResizedCrop(50, scale=(0.8, 1)),
        transforms.RandomErasing(scale=(0.02, 0.2)),
        # transforms.GaussianBlur(3, sigma=(0.01, 0.01)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip()
])
train_dataset = datasets.ImageFolder(
    "data/train_folder", transform=train_transform
)

validation_transform = transforms.Compose([
        # transforms.Grayscale(), 
        transforms.Resize(150),
        transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize(15),
])

validation_dataset = datasets.ImageFolder(
    "data/validation_folder", transform=validation_transform
)

kwargs = {"num_workers": 5, "pin_memory": True}
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, **kwargs
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=32, shuffle=False, **kwargs
)

cnt = 0
for i, _ in train_loader:
    cnt += 1
    if cnt > 10:
        break
    plt.imshow(i[0][0])
    plt.show()

# %%
def train(train_loader, optimizer, epoch, criterion):
    net.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = net(data)
        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(loader, criterion, dataset_name):
    net.eval()

    loss = 0
    correct = 0

    for data, target in loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            prediction = net(data)
            loss += criterion(prediction, target)

            correct += th.sum(prediction.argmax(dim=1) == target)

    loss /= len(loader.dataset)

    percentage_correct = 100.0 * correct / len(loader.dataset)

    print(
        dataset_name + ": {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(loader.dataset), percentage_correct
        ),
        flush=True
    )

    return loss.item(), percentage_correct.item()

def try_improove(acc):
    global best_act
    if acc <= best_act:
        return

    best_act = acc
    submission.make_submission(net, True, "test", test_transform, False)

    print("New best:", best_act)
    th.save(net, "data/resnet_sav.th")


# %%

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


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        # self.resnet = models.resnet18()
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1000, 3)
        )

    def forward(self, x):
        x = self.fc(self.resnet(x))
        return x


net = Resnet().cuda()
criterion = nn.CrossEntropyLoss()
ep = 0
test_acc, train_acc = [], []

# %%

optimizer = torch.optim.Adam(
    net.parameters(), lr=3e-4
)

for e in range(1000):
    print(f"Training epoch #{ep}", flush=True)
    train(train_loader, optimizer, ep, criterion)
    train_acc.append(test(train_loader, criterion, "Train dataset"))
    test_acc.append(test(validation_loader, criterion, "Validation dataset"))
    try_improove(test_acc[-1][1])
    print("")

# %%

test(train_loader, criterion, "Train")
test(validation_loader, criterion, "Validation")

# %%
def ConfusionArrays(model, loader, criterion, dataset_name):
    model.eval()

    loss = 0
    correct = 0

    labels = []
    predictions = []

    for data, target in tqdm(loader):
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            prediction = model(data)
            
            for i in target:
                labels.append(i.item())
            for i in prediction:
                predictions.append(th.argmax(i).item())
    
    return labels, predictions

labels, predicted = ConfusionArrays(net, validation_loader, criterion, "Validation")
plt.imshow(confusion_matrix(labels, predicted), cmap='gray')
plt.show()
print(confusion_matrix(labels, predicted))
plt.plot([b for a, b in train_acc], label="training")
plt.plot([b for a, b in test_acc], label="testing")
plt.legend()
plt.show()

# %%
th.save(net, "resnet_sav.th")
# %%
net = th.load("data/resnet_sav.th").cuda()
# %%
submission.make_submission(net, True, "validation", True)
# %%
