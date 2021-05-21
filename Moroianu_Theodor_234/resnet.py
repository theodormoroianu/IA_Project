
# %%

# Import required packages.
import torch as th
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from torchvision import models, datasets, transforms
import torch as th
import torchvision.transforms.functional as T
from positional_encodings import PositionalEncodingPermute2D
import time

# Check if cuda is enabled.
dev = torch.device('cpu')
if th.cuda.is_available():
    dev = torch.device('cuda')

print("Device:", dev)

# Best accuracy on validation so far.
best_act = 0.

# %%

# Power of color jitter, and resize size.
TRANF_POWER = 0.2
RESIZE = 200

# Path of the data.
datapath = "data"

# Training image transformations.
train_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        # transforms.Grayscale(),
        # transforms.ColorJitter(2 * TRANF_POWER, 2 * TRANF_POWER, TRANF_POWER, TRANF_POWER),
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=20),
        transforms.RandomCrop(RESIZE, padding=10),
        # transforms.RandomAutocontrast(),
        # transforms.RandomAdjustSharpness(0.95),
        # transforms.RandomResizedCrop(RESIZE, scale=(0.8, 1)),
        # transforms.RandomEqualize(),
        # PEnc(),
        # transforms.RandomErasing(scale=(0.02, 0.2)),
        # transforms.GaussianBlur(3, sigma=(0.01, 0.01)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip()
])

# Defining datasets / dataloaders.
train_dataset = datasets.ImageFolder(
    datapath + "/train_folder", transform=train_transform
)

validation_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        # PEnc(),
])

validation_dataset = datasets.ImageFolder(
    datapath + "/validation_folder", transform=validation_transform
)

test_dataset = datasets.ImageFolder(
    datapath + "/test_folder", transform=validation_transform
)

# For faster loading time.
kwargs = {
    "num_workers": 5,
    "pin_memory": True
}
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, **kwargs
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=64, shuffle=False, **kwargs
)

# Show a few images.
nr_shown = 0
for img, _ in train_dataset:
    nr_shown += 1
    if nr_shown > 5:
        break
    plt.imshow(img[0])
    plt.show()

# %%

# Makes a new submission.
# If contains_label is true, then tries to check accuracy.
def make_submission(dataset, contains_labels=False):
    with th.no_grad():
        net.eval()
        acc = 0
        fout = open(datapath + "/submission.csv", "w")
        fout.write("id,label\n")
        for id, (img, label) in enumerate(dataset):
            result = net(img.to(dev).view(1, img.shape[0], img.shape[1], img.shape[2]))

            predicted = result.argmax(dim=1)[0].item()

            if contains_labels:
                acc += (1 if predicted == label else 0)
            
            filename = dataset.imgs[id][0].split('/')[3]
            fout.write(filename + "," + str(predicted) + "\n")

        fout.close()

        if contains_labels:
            nr = len(dataset)
            print(f"Accuracy: {round(acc / nr * 100, 2)}%")

# Trains the model one epoch.
def train(train_loader, optimizer, epoch, criterion):
    net.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.to(dev)
        target = target.to(dev)

        optimizer.zero_grad()

        prediction = net(data)
        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")

# Returns the accuracy and loss of the model for the given dataset.
def test(loader, criterion, dataset_name):
    net.eval()

    loss = 0
    correct = 0

    for data, target in loader:
        with torch.no_grad():
            data = data.to(dev)
            target = target.to(dev)

            prediction = net(data)
            loss += criterion(prediction, target)

            correct += th.sum(prediction.argmax(dim=1) == target)

    loss /= len(loader)

    percentage_correct = 100.0 * correct / len(loader.dataset)

    print(
        dataset_name + ": {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(loader.dataset), percentage_correct
        ),
        flush=True
    )

    return loss.item(), percentage_correct.item()

# Tries to improve the solution.
def try_improve(acc):
    global best_act
    if acc <= best_act:
        return

    best_act = acc
    make_submission(test_dataset, False)

    print("New best:", best_act)
    th.save(net, "data/resnet_sav.th")

test_acc, train_acc = [], []
time_taken = []

# %%

# Resnet network, with an additional classifier.
class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1000, 3)
        )

    def forward(self, x):
        x = self.fc(self.resnet(x))
        return x

# Create the net and criterion.
net = Resnet().to(dev)
criterion = nn.CrossEntropyLoss()
ep = 0
# Disable gradients of the pretrained network.
net.resnet.requires_grad = False

# %%

# Enable gradients of the pretrained network.
net.resnet.requires_grad = True


# %%

# Define the optimizer.
optimizer = torch.optim.Adam(
    net.parameters(), lr=1e-4
)

# Train the network.
for e in range(1000):
    print(f"Training epoch #{ep}", flush=True)
    before = time.time()
    train(train_loader, optimizer, ep, criterion)
    after = time.time()
    train_acc.append(test(train_loader, criterion, "Train dataset"))
    test_acc.append(test(validation_loader, criterion, "Validation dataset"))
    time_taken.append(after - before)
    try_improve(test_acc[-1][1])
    print("")
    ep += 1

# %%

# Compute confusion Arrays.
def ConfusionArrays(model, loader, criterion, dataset_name):
    model.eval()

    loss = 0
    correct = 0

    labels = []
    predictions = []

    for data, target in tqdm(loader):
        with torch.no_grad():
            data = data.to(dev)
            target = target.to(dev)

            prediction = model(data)
            
            for i in target:
                labels.append(i.item())
            for i in prediction:
                predictions.append(th.argmax(i).item())
    
    return labels, predictions

# Display various stuff.
labels, predicted = ConfusionArrays(net, validation_loader, criterion, "Validation")
plt.imshow(confusion_matrix(labels, predicted), cmap='gray')
plt.show()
print(confusion_matrix(labels, predicted))
plt.plot([b for a, b in train_acc], label="training")
plt.plot([b for a, b in test_acc], label="testing")
plt.legend()
plt.show()

#%%
# Display train loss and validation loss for each epoch as scatter points.
train_loss = [i[0] for i in train_acc]
val_loss = [i[0] for i in test_acc]

fig, ax = plt.subplots()

ax.scatter(train_loss, val_loss, s=time_taken)

for i in range(8):
    ax.annotate(str(i + 1), (train_loss[i] + 0.003, val_loss[i] + 0.03))

ax.set_xlabel("Train Loss")
ax.set_ylabel("Validation Loss")
# %%
