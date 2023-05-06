class Perceptron:
    def __init__(self, theta, feature_count, epoch):
        self.weights = [0 for i in range feature_count]
        self.theta = theta
        self.epoch = epoch

    def fit(self, X, y):
        for _ in range(epoch):
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
