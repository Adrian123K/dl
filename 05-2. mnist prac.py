# import sys, os
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
#
# (x_train,t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

# 2.
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mgimg
#
# img = Image.open('아이린.jpg')
# img_pixel = np.array(img)
# plt.imshow(img_pixel)

import pickle

def init_network():
    with open("D:/dl/sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network

network = init_network()

print(network["W1"].shape)
print(network["W2"].shape)
print(network["W3"].shape)
print(network["b1"].shape)
print(network["b2"].shape)
print(network["b3"].shape)

