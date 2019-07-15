import numpy as np

def bound_nonuniform_sampler(*args):
    x = np.random.randn(*args)*0.1 + 0.5
    x[x < 0] = -x[x < 0]
    x[x > 1] = x[x > 1] - 1
    x[x < 0] = -x[x < 0]
    return x

def uniform_sampler(*args):
    x = np.random.rand(*args)
    x = (x - 0.5) * 3
    return x