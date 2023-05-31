import numpy as np
from torchvision import datasets
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch import nn


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 32, stride=1, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 64*7*7=3136 (Comment out last 2 lines and print x.size() in forward)
            nn.Flatten(),
            nn.Linear(3136, 2)
        )
        # TODO: Look into why the reverse of the encoder doesn't generate 28*28
        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 3136),
            nn.Unflatten(-1, (64, 7, 7)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),                
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=0),                
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, stride=1, kernel_size=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        #print(x.size())
        x = self.decoder(x)
        #print(x.size())
        return x


RANDOM_SEED = 1
BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(RANDOM_SEED)

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

model = AutoEncoder()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def train():
    for epoch in range(NUM_EPOCHS):
        model.train()
        for features, targets in train_loader:
            features = features.to(DEVICE)
            # Call checks and forward
            logits = model(features)
            #if epoch == NUM_EPOCHS-1:
            #    plt.imshow(features[0].reshape(28, 28), cmap='gray')
            #    plt.show()
            #    plt.imshow(logits[0].reshape(28, 28).detach().numpy(), cmap='gray')
            #    plt.show()
            # Calculate Loss
            cost = F.mse_loss(logits, features)
            # Reset gradients
            optimizer.zero_grad()
            # Calculate gradient
            cost.backward()
            # Update weights
            optimizer.step()
            print(f'Cost/Loss: {cost}')
train()

