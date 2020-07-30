import numpy as np
x = np.array([-1,0,0,-1,1,0,-1,0,1,-1,1,1]).reshape(4,3)
w = np.array([0.3,0.4,0.1]).reshape(3,1)
x, w

print(x.shape)
print(w.shape)

x[0]*w.T
np.sum(x[0]*w.T)