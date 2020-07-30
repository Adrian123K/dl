import numpy as np

def pcn(x, t):
    w = np.random.randn(len(x), 1)

    def step_func(x):
        y = np.array(x > 0)
        return y.astype(np.int)

    def predict(x, w):
        a = np.sum(x * w.T)
        return step_func(a)

    lr = 0.05
    while 1:
        cnt = 0
        for i in range(len(x)):
            cost = t[i] - predict(x[i], w)
            if cost != 0:
                w = w + np.array([lr * x[i] * cost]).T
            elif cost:
                continue
            cnt += cost
        if cnt == 0:
            break
    return w
