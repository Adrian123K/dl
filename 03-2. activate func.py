import matplotlib.pyplot as plt
import numpy as np

x_data = np.array([-1,0,1])
def step_function(x):
    y = x>0
    return y.astype(int)
print(step_function(x_data))

import numpy as np
def softmax(x):
    mx = np.max(x)
    exp_x = np.exp(x-mx)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

a = np.array([1010,1000,990])
softmax(a)