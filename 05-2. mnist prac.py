# # # import sys, os
# # # sys.path.append(os.pardir)
# # # from dataset.mnist import load_mnist
# # #
# # # (x_train,t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# # #
# # # print(x_train.shape)
# # # print(t_train.shape)
# # # print(x_test.shape)
# # # print(t_test.shape)
# #
# # # 2.
# # # from PIL import Image
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import matplotlib.image as mgimg
# # #
# # # img = Image.open('아이린.jpg')
# # # img_pixel = np.array(img)
# # # plt.imshow(img_pixel)
# #
# # import pickle
# #
# # def init_network():
# #     with open("D:/dl/sample_weight.pkl","rb") as f:
# #         network = pickle.load(f)
# #     return network
# #
# # network = init_network()
# #
# # print(network["W1"].shape)
# # print(network["W2"].shape)
# # print(network["W3"].shape)
# # print(network["b1"].shape)
# # print(network["b2"].shape)
# # print(network["b3"].shape)
#
# import sys, os
# import pickle
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
# from common.functions import sigmoid, softmax, np
#
# def get_data():
#     (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False )
#     return x_test, t_test
#
# def init_network():
#     with open('d:/dl/sample_weight.pkl','rb') as f:
#         network=pickle.load(f)
#
#     return network
#
# def predict(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#
#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = softmax(a3)
#
#     return y
#
# x, t = get_data()
# network = init_network()
#
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)
#     if p == t[i]:
#         accuracy_cnt += 1
# # y = predict(network,x[0])
# print(t[0]) # 7
# print(y)
# print("Accuracy:"+str(float(accuracy_cnt)/len(x))) # 0.9352

import sys, os
import pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax, np

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False )
    return x_test, t_test

def init_network():
    with open('d:/dl/sample_weight.pkl','rb') as f:
        network=pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy = []
batch_size = 100
for i in range(0,len(x),batch_size):
    y = predict(network,x[i:i+batch_size])
    accuracy.append(sum(np.argmax(y, axis=1)==t[i:i+batch_size])/100)
print(accuracy)