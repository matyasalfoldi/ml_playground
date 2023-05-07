import numpy as np
import torch

class Perceptron:
    def __init__(self, theta, feature_count, epochs):
        self.weights = [0 for i in range(feature_count)]
        self.theta = theta
        self.epochs = epochs

    def fit(self, X, y):
        for _ in range(epochs):
            for x, label in zip(X, y):
                pred = self.predict(x)
                if pred != label:
                    error = label - pred
                    for i, x_i in enumerate(x):
                        self.weights[i] += error*x_i
            
    def predict(self, x):
        net_input = -self.theta
        for feature, weight in zip(x, self.weights):
            net_input += feature*weight
        return 1 if net_input > 0 else 0


class Perceptron_np:
    def __init__(self, feature_count, epochs):
        self.feature_count = feature_count
        self.weights = np.zeros(feature_count, dtype=np.float16)
        self.bias = np.zeros(1, dtype=np.float16)
        self.epochs = epochs

    def _net_input(self, x):
        return x.dot(self.weights) + self.bias

    def fit(self, X, y):
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                pred = self.predict(X[i])
                if pred != y[i]:
                    error = y[i] - pred
                    self.weights += error
                    self.bias += error
                
    def predict(self, x):
        net_input = self._net_input(x)
        return 1 if net_input > 0 else 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
class Perceptron_pytorch:
    def __init__(self, feature_count, epochs):
        self.feature_count = feature_count
        self.epochs = epochs
        self.weights = torch.zeros(
            feature_count,
            1,
            dtype=torch.float32,
            device=device
        )
        self.bias = torch.zeros(
            1,
            dtype=torch.float32,
            device=device
        )

    # does the prediction
    def forward(self, x):
        net_input = torch.matmul(x, self.weights) + self.bias
        pred = torch.where(net_input > 0., 1., 0.)
        return pred
    
    # returns errors
    def backward(self, x, y):
        # Needed for matrix mul to create a 1xfeature_count matrix
        pred = self.forward(x.reshape(1, self.feature_count))
        errors = y - pred
        return errors
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                errors = self.backward(X[i], y[i]).reshape(-1)
                # Create a column vector from row vector
                self.weights += (errors * X[i]).reshape(self.feature_count, 1)
                self.bias += errors
                
    def evaluate(self, X, y):
        # Create row vector
        pred = self.forward(X).reshape(-1)
        acc = torch.sum(pred == y).float() / y.shape[0]
        return acc


if __name__ == '__main__':
    data = np.genfromtxt('perceptron_toydata.txt', delimiter='\t')
    X, y = data[:, :2], data[:, 2]
    y = y.astype(np.int32)

    print('label mix:', np.bincount(y))

    shuffle_idx = np.arange(y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)
    X, y = X[shuffle_idx], y[shuffle_idx]

    X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
    y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

    # Normalize
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    print(f'mean: {mu}')
    print(f'std: {sigma}')
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    """
    ppn = Perceptron_np(feature_count=2, epochs=5)
    ppn.fit(X_train, y_train)

    y_pred = []
    for x in X_test:
        y_pred.append(ppn.predict(x))

    print(np.sum(np.asarray(y_pred) == y_test) / y_test.shape[0])
    """
    ppn = Perceptron_pytorch(2, 5)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    ppn.train(X_train_tensor, y_train_tensor)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

    test_acc = ppn.evaluate(X_test_tensor, y_test_tensor)
    print('Test set accuracy: %.2f%%' % (test_acc*100))
