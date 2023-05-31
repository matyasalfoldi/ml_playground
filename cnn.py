import numpy as np
from torchvision import datasets
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

RANDOM_SEED = 1
BATCH_SIZE = 256
NUM_EPOCHS = 50
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

resize = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        # Normalizes to 0-1 values
        torchvision.transforms.ToTensor()
    ]
)

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=resize,
    download=True
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=resize
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)


class LeNet5(torch.nn.Module):
    def __init__(self, num_classes, black_and_white=False):
        super().__init__()
        self.black_and_white = black_and_white

        if self.black_and_white:
            in_ch = 1
        else:
            in_ch = 3
        # Calculating padding size for same sized input/output:
        # p = (kernel_size-1)/2
        # output width = (input_width - kernel_width)/stride + 1
        # Here in Conv2d calls stride=1
        self.network = torch.nn.Sequential(
            # 32-5+1=28
            # 6 * 28*28 feature map
            # we can do padding=2 for example to add padding
            torch.nn.Conv2d(in_ch, 6, kernel_size=5),
            # LeNet 5 uses Tanh activation function
            torch.nn.Tanh(),
            # 2*2 so 28*28 is reduced to 14*14, because of default stride 2
            torch.nn.MaxPool2d(kernel_size=2),
            # 14-5+1=10
            # 6 * 10*10 feature map
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            # 2*2 so 10*10 is reduced to 5*5
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16*5*5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.network(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

def train():
    for epoch in range(NUM_EPOCHS):
        model.train()
        for features, targets in train_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            # Call checks and forward
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            # Calculate Loss
            cost = F.cross_entropy(logits, targets)
            # Reset gradients
            optimizer.zero_grad()
            # Calculate gradient
            cost.backward()
            # Update weights
            optimizer.step()


def evaluate():
    model.eval()
    acc = 0
    count = 0
    for features, targets in test_loader:
        count += 1
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        acc += torch.sum(torch.argmax(probs, dim=1) == targets.view(-1)) / targets.size(0)
    acc /= count
    print(f'{acc=}')

torch.manual_seed(RANDOM_SEED)
model = LeNet5(10, True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
model.to(DEVICE)
import os
model_path = 'save/model.pt'
optimizer_path = 'save/optimizer.pt'
if os.path.exists(model_path):
    print('Loading model/optimizer')
    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))
else:
    print('Training')
    train()
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
evaluate()

