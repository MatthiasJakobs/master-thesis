import torch
import numpy as np

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

def transform_2d_point(A, x):
    # point has shape (2,), so expansion is needed
    x = np.expand_dims(x, axis=-1)
    
    y = transform(A, x)
    y = np.squeeze(y)

    return y

def transform_pose(A, pose, inverse=False):
    new_pose = np.empty(pose.shape)

    if inverse:
        A = np.linalg.inv(A)

    for i, (x,y) in enumerate(pose):
        transformed_point = transform_2d_point(A, np.array([x, y]))
        new_pose[i] = transformed_point
    print(new_pose)
    return new_pose

def transform(A, x):
    dim, n = x.shape
    y = np.ones((dim+1, n))
    y[0:dim, :] = x[0:dim, :]
 
    return np.dot(A, y)[0:dim]

def scale(mat, x, y):
    t = np.eye(3)
    t[0,0] *= x
    t[1,1] *= y

    return np.dot(t, mat)

def flip(mat, a, d):
    t = np.eye(3)
    t[0,0] = a
    t[1,1] = d

    return np.dot(t, mat)

def flip_h(mat):
    return flip(mat, -1, 1)

def translate(mat, x, y):
    t = np.eye(3)
    t[0,2] = x
    t[1,2] = y

    return np.dot(t, mat)

def superflatten(array):
    return array.flatten()[0]

def get_valid_joints(pose):
    valid = pose > 1e-6
    valid_sum = np.sum(np.apply_along_axis(np.all, axis=2, arr=valid.cpu()), 1)
    valid_sum_tensor = torch.from_numpy(valid_sum)
    if torch.cuda.is_available():
        valid_sum_tensor.to(torch.cuda.current_device())
        return valid.float(), valid_sum_tensor.cuda().float()
    return valid.float(), valid_sum_tensor.float()

