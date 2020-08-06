import sys, os

sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error, relu
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W1 = np.random.randn(4, 5)
        self.W2 = np.random.randn(5, 4)
        self.W3 = np.random.randn(4, 5)
        self.W4 = np.random.randn(5, 2)

    def predict(self, x):
        W1, W2, W3, W4 = self.W1, self.W2, self.W3, self.W4
        s1 = np.dot(x, W1)
        r1 = relu(s1)
        s2 = np.dot(r1, W2)
        r2 = relu(s2)
        s3 = np.dot(r2, W3)
        r3 = relu(s3)
        s4 = np.dot(r3, W4)
        y = softmax(s4)
        return y

    def loss(self, x, t):
        z = self.predict(x)
        loss = cross_entropy_error(z, t)

        return loss

    def accuracy(self, x, t):
        z = self.predict(x)
        y = np.argmax(z)
        t = np.argmax(t)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


x = np.array([0.4, 0.1, 0.8, 0.9])
t = np.array([0, 1])
net = simpleNet()
print(net.accuracy(x, t))

