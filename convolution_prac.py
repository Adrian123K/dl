# class Convolution:
#     def __init__(self, W, b, stride=1, pad=0):
#         self.W = W
#         self.b = b
#         self.stride = stride
#         self.pad = pad
#
#     def forward(self, x):
#         FN, C, FH, FW = self.W.shape
#         N, C, H, W = x.shape
#         out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
#         out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)
#
#         col = im2col(x, FH, FW, self.stride, self.pad)
#         col_W = self.W.reshape(FN, -1).T
#         out = np.dot(col, col_W) + self.b
#
#         out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
#
#         return out
#
# import numpy as np
#
# x1 = np.random.randn(1,3,28,28)
#
# filter = np.random.randn(10,3,5,5)
# b1 = 1
#
# conv = Convolution(filter, b1)
# fe_map = conv.forward(x1)
# print(f'fe_map의 shape는 {fe_map.shape}')

import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

x = np.array([21,8,8,12,12,19,9,7,8,10,4,3,18,12,9,10]).reshape(4,4)

print(x.ndim) # 2
print(x.shape) # (4,4)

# 위의 2차원 행렬을 im2col의 입력값으로 넣을 수 있도록 4차원 행렬로 변환

x2 = x.reshape(1,1,4,4)

print(x2.shape)
print(x2.ndim)

print(x2)
print(im2col(x2, 2, 2, stride=2, pad=0))