import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 256
epochs = 10
lr = 0.1
seed = 123

num_features = 784 # 28*28
num_classes = 10

class SoftmaxRegression(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

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
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

model = SoftmaxRegression(num_features=num_features, num_classes=num_classes).to(device)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


for epoch in range(epochs):
    for features, targets in train_loader:
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)

        logits, probs = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()

        optimizer.step()
        """
        acc = torch.sum(torch.argmax(probs, dim=1) == targets.view(-1)) / targets.size(0)
        print(f'{acc=}')
        """

for features, targets in test_loader:
    features = features.view(-1, 28*28).to(device)
    targets = targets.to(device)
    _, probs = model(features)
    acc = torch.sum(torch.argmax(probs, dim=1) == targets.view(-1)) / targets.size(0)
    print(f'{acc=}')
