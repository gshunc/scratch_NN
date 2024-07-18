import random
import numpy as np
import pandas as pd


class Network():
    def __init__(self, dataset: np.Array, alpha: float = 0.1, test_ratio: float = 0.2):
        self.alpha = alpha

        dataset.shuffle()
        
        split_point = test_ratio * (len(dataset) - 1)
        test_data = dataset[:split_point]
        train_data = dataset[split_point:]

        self.X_train = train_data[1:].T
        self.y_train = train_data[0]

        self.X_test = train_data[1:].T
        self.y_train = train_data[0]

        self.w1, self.w2, self.b1, self.b2 = self.initialize_params()

    def initialize_params(self):
        w1 = np.random.rand(10,784)
        w2 = np.random.rand(10,784)
        b1 = np.random.rand(10,1)
        b2 = np.random.rand(10,1)

        return w1, w2, b1, b2

    def relu(self, x):
        return x if x > 0 else 0
    
    def d_relu(self, x):
        return 1 if x > 0 else 0
    
    def softmax(self, vector):
        return np.exp(vector) / np.sum(np.exp(vector))
    
    def one_hot(self, y):
        one_hot_y = np.array((y.size(),y.max() + 1))
        one_hot_y[np.arange(y.size()), y] = 1
        return one_hot_y.T
    
    def forward(self, X):
        z1 = self.w1.dot(X) + self.b1
        a1 = self.relu(z1)

        z2 = self.w2.dot(a1) + self.b2
        a2 = self.softmax(z2)

        return z1, a1, z2, a2
    
    def backward(self):
        