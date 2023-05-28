import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

RANDOM_SEED = 1
BATCH_SIZE = 100
NUM_EPOCHS = 10
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.MNIST(
    root='data', 
    train=True, 
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root='data', 
    train=False, 
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

class MultilayerPeceptron(torch.nn.Module):

    def __init__(self, num_features, num_hidden, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.hidden = torch.nn.Linear(num_features, num_hidden)
        self.linear_out = torch.nn.Linear(num_hidden, num_classes)
        
    def forward(self, x):
        out = self.hidden(x)
        # Activation function for hidden layer
        out = torch.sigmoid(out)
        logits = self.linear_out(out)
        # Activation function for out layer
        probs = torch.softmax(logits, dim=1)
        return logits, probs

torch.manual_seed(RANDOM_SEED)
model = MultilayerPeceptron(
    num_features=28*28,
    num_hidden=100,
    num_classes=10
)

model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(NUM_EPOCHS):
    model.train()
    for features, targets in train_loader:
        features = features.view(-1, 28*28).to(DEVICE)
        targets = targets.to(DEVICE)
        # Call checks and forward
        logits, probs = model(features)
        # Calculate Loss
        cost = F.cross_entropy(logits, targets)
        # Reset gradients
        optimizer.zero_grad()
        # Calculate gradient
        cost.backward()
        # Update weights
        optimizer.step()


for features, targets in test_loader:
    features = features.view(-1, 28*28).to(DEVICE)
    targets = targets.to(DEVICE)
    _, probs = model(features)
    acc = torch.sum(torch.argmax(probs, dim=1) == targets.view(-1)) / targets.size(0)
    print(f'{acc=}')