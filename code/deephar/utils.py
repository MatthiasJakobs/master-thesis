import torch
from numpy import linspace, empty

def spatial_softmax(x):
    maximum, _ = torch.max(x, 1, keepdim=True)
    maximum, _ = torch.max(maximum, 2, keepdim=True)

    e = torch.exp(x - maximum)
    s = torch.sum(e, (1,2), keepdim=True)
    return e / s

def linspace_2d(rows, cols, dim):
    x = empty((rows, cols))

    if dim == 0:
        space = linspace(0.0, 1.0, cols)

        for i in range(rows):
            x[i] = space

    else:
        space = linspace(0.0, 1.0, rows)

        for i in range(cols):
            x[:,i] = space

    return x



