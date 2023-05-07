import numpy as np
import matplotlib.pyplot as plt

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
        self.weights = np.zeros((feature_count, 1), dtype=np.float16)
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

    ppn = Perceptron_np(feature_count=2, epochs=5)
    ppn.fit(X_train, y_train)

    y_pred = []
    for x in X_test:
        y_pred.append(ppn.predict(x))

    print(np.sum(np.asarray(y_pred) == y_test) / y_test.shape[0])
