# # import sys, os
# # import pickle
# # sys.path.append(os.pardir)
# # from dataset.mnist import load_mnist
# # from common.functions import sigmoid, softmax, np
# #
# # def get_data():
# #     (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False )
# #     return x_test, t_test
# #
# # def init_network():
# #     with open('d:/dl/sample_weight.pkl','rb') as f:
# #         network=pickle.load(f)
# #
# #     return network
# #
# # def predict(network, x):
# #     W1, W2, W3 = network['W1'], network['W2'], network['W3']
# #     b1, b2, b3 = network['b1'], network['b2'], network['b3']
# #
# #     a1 = np.dot(x, W1) + b1
# #     z1 = sigmoid(a1)
# #     a2 = np.dot(z1, W2) + b2
# #     z2 = sigmoid(a2)
# #     a3 = np.dot(z2, W3) + b3
# #     y = softmax(a3)
# #
# #     return y
# #
# # x, t = get_data()
# # network = init_network()
# #
# # batch_size = 100
# #
# # for i in range(0,len(x),batch_size):
# #     batch_mask = np.random.choice(len(x),batch_size)
# #     x_batch = x[batch_mask]
# #     y = predict(network, x_batch)
# #     print(y)
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
# batch_size = 100
#
# for i in range(0,len(x),batch_size):
#     cnt = 0
#     batch_mask = np.random.choice(len(x),batch_size)
#     x_batch = x[batch_mask]
#     t_batch = t[batch_mask]
#     y_batch = predict(network, x_batch)
#     cnt += sum(np.argmax(y_batch, axis=1) == t_batch)
#     print(cnt)
