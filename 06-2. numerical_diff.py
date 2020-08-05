import numpy  as  np

def f(x):
    return x[0] ** 2 + x[1] ** 2

def loss_func(x):
    return x[0] ** 2 + x[1] ** 2

def numerical_gradient(f, x):
    h = 1e-04
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i]

        x[i] = tmp_val + h
        fxh1 = f(x)

        x[i] = tmp_val - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)

        x[i] = tmp_val

    return grad

x = np.array([3.0, 4.0])
print(numerical_gradient(loss_func, x))

