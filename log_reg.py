import numpy as np
import torch
import torch.nn.functional as F

# For Binary Classification
class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        # Will return the probability
        return torch.sigmoid(self.linear(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.genfromtxt('toydata.txt', delimiter='\t')
X = data[:, :2].astype(np.float32)
y = data[:, 2].astype(np.int64)
# Set random seed, for reproducibility
np.random.seed(123)
idx = np.arange(y.shape[0])
np.random.shuffle(idx)
X_test, y_test = X[idx[:25]], y[idx[:25]]
X_train, y_train = X[idx[25:]], y[idx[25:]]
# Normalize the input
mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train, X_test = (X_train - mu) / std, (X_test - mu) / std

model = LogisticRegression(num_features=2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
epochs = 500

def comp_accuracy(label_var, pred_probas):
    pred_labels = torch.where((pred_probas > 0.5), 1, 0).view(-1)
    acc = torch.sum(pred_labels == label_var.view(-1)).float() / label_var.size(0)
    return acc

for epoch in range(epochs):
    # Calls forward in the background, calculates y^
    out = model(X_train_t)
    cost = F.binary_cross_entropy(out, y_train_t, reduction='sum')
    optimizer.zero_grad()
    # Calculate grad-s
    cost.backward()

    # Update weights
    optimizer.step()

    # Check how the training went in the epoch
    pred_probabilities = model(X_train_t)
    predictions = torch.where((pred_probabilities > 0.5), 1, 0).view(-1)
    accuracy = torch.sum(predictions == y_train_t.view(-1)).float() / y_train_t.size(0)
    print(f'Accuracy after epoch: {epoch}={accuracy}')
