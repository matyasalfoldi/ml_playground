import numpy as np

class EmptyPQ(Exception):
    pass

class KIterablePriorityQueue:
    def __init__(self, k):
        self.k = k
        self.storage = []

    def front(self):
        if self.storage:
            return self.storage[0]
        else:
            raise EmptyPQ

    def put(self, x):
        self.storage.append(x)
        self.storage.sort(key=lambda x: x[1], reverse=True)

    def pop(self):
        return self.storage.pop(0)

    def size(self):
        return len(self.storage)

    def __getitem__(self, i):
        return self.storage[i]
    
    def __len__(self):
        return self.size()

# K nearest neighbor for 2 features
class KNearestNeighbor:
    def __init__(self, k):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def _get_label(self, x):
        return self.y[np.where(self.X == x)]

    def predict(self, inp):
        k_nearest = KIterablePriorityQueue(k=self.k)
        for x in X:
            dist = sqrt((x[0] - inp[0]) ** 2 + (x[1] - inp[1]) ** 2)
            if len(k_nearest) < 4:
                k_nearest.put((x, dist, self._get_label(x)))
            elif k_nearest.front() > dist:
                k_nearest.pop()
                k_nearest.put((x, dist, self._get_label(x)))
        classes = np.unique(self.y)
        argmax = -1
        argmax_label = -1
        for cls in classes:
            s = 0
            for near in k_nearest:
                s += 1 if cls == near[2] else 0
            if argmax < s:
                argmax = s
                argmax_label = cls
