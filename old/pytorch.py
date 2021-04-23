# %%

# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

import matplotlib.pyplot as plt
from numpy import random
import torch.nn as nn
from torch import optim
import torch as th
from torchvision import datasets, transforms
import os
import numpy as np
from ray import tune
from functools import partial
import torch.nn.functional as F
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from lr_reduce import ReduceLROnPlateau
from ray.tune.schedulers import ASHAScheduler

# %%
# normalize = transforms.Normalize((0.52,), (0.34,))
def load_data(data_dir=None):
    TRANF_POWER = 0.1

    train_transform = transforms.Compose([
            transforms.Grayscale(),
            # transforms.ColorJitter(2 * TRANF_POWER, 2 * TRANF_POWER, TRANF_POWER, TRANF_POWER),
            transforms.ToTensor(),
            # normalize,
            # transforms.RandomCrop(50, padding=4),
            # transforms.RandomRotation(degrees=15),
            # transforms.GaussianBlur(3, sigma=(0.01, 0.01)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip()
    ])
    train_dataset = datasets.ImageFolder(
        data_dir + "/train_folder", transform=train_transform
    )

    validation_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            # normalize,
    ])
    validation_dataset = datasets.ImageFolder(
        data_dir + "/validation_folder", transform=validation_transform
    )
    test_dataset = datasets.ImageFolder(
        data_dir + "/test_folder", transform=validation_transform
    )
    
    return train_dataset, validation_dataset, test_dataset

# %%
class ConvUnit(nn.Module):
    def __init__(self, in_f, out_f, ker=3, dropout=0.05, max_pull=False):
        super().__init__()
        if max_pull == False:
            self.net = nn.Sequential (
                nn.Conv2d(in_f, out_f, kernel_size=ker, padding=ker//2),
                nn.BatchNorm2d(out_f),
                nn.Dropout2d(dropout),
                nn.ReLU()
            )
        else:
            self.net = nn.Sequential (
                nn.Conv2d(in_f, out_f, kernel_size=ker, padding=ker//2, stride=2),
                nn.BatchNorm2d(out_f),
                nn.Dropout2d(dropout),
                nn.ReLU()
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



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            ConvUnit(1, 32, ker=config['ker1']),
            ConvUnit(32, config['d1'], ker=config['ker1'], max_pull=True, dropout=config['drop1']),
            ConvUnit(config['d1'], config['d2'], ker=3, max_pull=True, dropout=config['drop1']),

            nn.Flatten(),

            FcUnit(config['d2'] * 13 * 13, config['fc1'], dropout=config['drop2']),
            FcUnit(config['fc1'], config['fc2'], dropout=config['drop3']),
            FcUnit(config['fc2'], 3, dropout=0.)
        )

    def forward(self, x):
        return self.net(x)

# %%
def train_model(config, checkpoint_dir=None, data_dir=None):
    net = Model(config)

    device = "cpu"
    if th.cuda.is_available():
        device = "cuda:0"
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), 1e-4)
    # optimizer = th.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')

    if checkpoint_dir:
        model_state, optimizer_state = th.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, valset, _ = load_data(data_dir)
    
    trainloader = th.utils.data.DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = th.utils.data.DataLoader(
        valset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.detach().cpu().numpy(), epoch)

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with th.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = th.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            th.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

# %%
def test_accuracy(net, device="cpu"):
    _, _, testset = load_data()

    testloader = th.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with th.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = th.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
# %%

config = {
    "ker1": tune.choice([3, 5, 7]),
    "d1": tune.choice([32, 64, 128]),
    "d2": tune.choice([64, 128, 256]),
    # "d3": tune.sample_from(lambda _: 2**np.random.randint(6, 8)),
    "fc1": tune.choice([100, 500, 1000]),
    "fc2": tune.choice([100, 250, 500]),
    "drop1": tune.sample_from(lambda _: np.random.uniform(0., 0.2)),
    "drop2": tune.sample_from(lambda _: np.random.uniform(0.3, 0.7)),
    "drop3": tune.sample_from(lambda _: np.random.uniform(0., 0.7)),
    # "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64])
}
# %%
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10, # max nr of epochs
    grace_period=2,
    reduction_factor=2)

reporter = CLIReporter(
    parameter_columns=["ker1", "d1", "d2", "fc1", "fc2", "drop1", "drop2", "drop3", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"])

data_dir = os.path.abspath("./")
print("Datadir: " + data_dir)

result = tune.run(
    partial(train_model, data_dir=data_dir),
    resources_per_trial={"cpu": 2, "gpu": 1},
    config=config,
    num_samples=100,
    # gpus_per_trial=0.5,
    scheduler=scheduler,
    progress_reporter=reporter)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))

# %%
