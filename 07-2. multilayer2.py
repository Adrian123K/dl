# import sys, os
#
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
# from common.functions import *
# from common.gradient import numerical_gradient
#
# def get_data():
#     (x_train, t_train), (x_test, t_test) = \
#         load_mnist(normalize=True, flatten=True, one_hot_label=True)
#     return x_train, t_train
#
# class TwoLayerNet:
#     def __init__(self):
#         self.params = {}
#         self.params['W1'] = 0.01 * np.random.randn(784, 50)
#         self.params['b1'] = np.zeros(50)
#         self.params['W2'] = 0.01 * np.random.randn(50, 10)
#         self.params['b2'] = np.zeros(10)
#
#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#
#         a1 = np.dot(x, W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2) + b2
#         y = softmax(a2)
#
#         return y
#
#     def softmax(self, a):
#         minus = a - np.max(a)
#         return np.exp(minus) / np.sum(np.exp(minus))
#
#     def cross_entropy_error(self, y, t):
#         delta = 1e-7
#         if y.ndim == 1:
#             t = t.reshape(1, t.size)
#             y = y.reshape(1, y.size)
#         batch_size = y.shape[0]
#         return -np.sum(t * np.log(y + delta)) / batch_size
#
#     def loss(self, x, t):
#         z = self.predict(x)
#         y = self.softmax(z)
#         return self.cross_entropy_error(y, t)
#
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y_hat = np.argmax(y, axis=1)
#         target = np.argmax(t, axis=1)
#         accuracy = np.sum(y_hat == target) / float(x.shape[0])
#
#         return accuracy
#
# net = TwoLayerNet()
# x, t = get_data()
#
# y = net.predict(x[:100])
# y_hat = np.argmax(y, axis=1)
# print(y_hat)
# # y = net.accuracy(x[:100],t[:100])
# # print(y)
#
# import numpy  as  np
#
#
# def numerical_gradient(f, x):
#     h = 0.0001
#     grad = np.zeros_like(x)
#
#     for i in range(x.size):
#         tmp_val = x[i]
#         x[i] = tmp_val + h
#         fxh1 = f(x)
#
#         x[i] = tmp_val - h
#         fxh2 = f(x)
#
#         grad[i] = (fxh1 - fxh2) / (2 * h)
#
#         x[i] = tmp_val
#
#     return grad
#
#
# def numerical_gradient(self, x, t):
#     loss_W = lambda W: self.loss(x, t)
#
#     grads = {}
#     grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#     grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#     grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#     grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import *
from common.gradient import numerical_gradient


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return x_train, t_train


class TwoLayerNet:
    def __init__(self):
        self.params = {}
        self.params['W1'] = 0.01 * np.random.randn(784, 50)
        self.params['b1'] = np.zeros(50)
        self.params['W2'] = 0.01 * np.random.randn(50, 10)
        self.params['b2'] = np.zeros(10)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def softmax(self, a):
        minus = a - np.max(a)
        return np.exp(minus) / np.sum(np.exp(minus))

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + delta)) / batch_size

    def loss(self, x, t):
        z = self.predict(x)
        y = self.softmax(z)
        return self.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y_hat = np.argmax(y, axis=1)
        target = np.argmax(t, axis=1)
        accuracy = np.sum(y_hat == target) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


net = TwoLayerNet()
x, t = get_data()
# batch_size = np.random.choice(len(x),100)
batch_size = 100
lr = 0.01

for i in range(10):
    x_batch = x[:batch_size]
    t_batch = t[:batch_size]
    grad = net.numerical_gradient(x_batch, t_batch)

    for key in ('W1','W2','b1','b2'):
        net.params[key] -= lr*grad[key]
    y = net.accuracy(x_batch, t_batch)
    print(y)
