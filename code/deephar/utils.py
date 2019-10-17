import torch
import numpy as np

from numpy import linspace, empty
from scipy.stats import multivariate_normal

def spatial_softmax(x):
    maximum, _ = torch.max(x, 1, keepdim=True)
    maximum, _ = torch.max(maximum, 2, keepdim=True)

    e = torch.exp(x - maximum)
    s = torch.sum(e, (1,2), keepdim=True)
    return e / s

def flip_lr_pose(pose):
    switches = [
        [0, 5],
        [1, 4],
        [2, 3],
        [10, 15],
        [11, 14],
        [12, 13]
    ]

    for (src ,dst) in switches:
        backup = pose[dst].clone()
        pose[dst] = pose[src]
        pose[src] = backup

    return pose

def linspace_2d(rows, cols, dim):

    if dim == 0:
        x = empty((rows, cols))

        space = linspace(0.0, 1.0, cols)

        for i in range(rows):
            x[i] = space

        return x

    else:
        x = empty((cols, rows))

        space = linspace(0.0, 1.0, rows)

        for i in range(cols):
            x[i] = space

        return x.T


def transform_2d_point(A, x, inverse=False):
    # point has shape (2,), so expansion is needed
    x = np.expand_dims(x, axis=-1)
    
    if inverse:
        if torch.cuda.is_available():
            A = np.linalg.inv(A.cpu())
        else:
            A = np.linalg.inv(A)
    
    y = transform(A, x)
    y = np.squeeze(y)

    return y

def transform_pose(A, pose, inverse=False):
    new_pose = np.empty(pose.shape)

    for i, (x,y) in enumerate(pose):
        transformed_point = transform_2d_point(A, np.array([x, y]), inverse=inverse)
        new_pose[i] = transformed_point

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

def get_valid_joints(pose, need_sum=True):
    valid = pose > 1e-6
    if not need_sum:
        if isinstance(pose, np.ndarray):
            return valid
        else:
            return valid.float()
    
    if isinstance(pose, np.ndarray):
        valid_sum = np.sum(np.apply_along_axis(np.all, axis=2, arr=valid), 1)
    else:
        valid_sum = np.sum(np.apply_along_axis(np.all, axis=2, arr=valid.cpu()), 1)
    valid_sum_tensor = torch.from_numpy(valid_sum)
    if torch.cuda.is_available():
        valid_sum_tensor.to(torch.cuda.current_device())
        return valid.float(), valid_sum_tensor.cuda().float()
    return valid.float(), valid_sum_tensor.float()

def create_heatmap(x, y, covariance, width=255, height=255):
    kernel = multivariate_normal(mean=(x,y), cov=np.eye(2)*covariance)

    xlim = (0, width)
    ylim = (0, height)
    xres = width
    yres = height
    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)

    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = kernel.pdf(xxyy)

    img = zz.reshape((yres,xres))
    return img

def get_bbox_from_pose(pose, bbox_offset=None):
    # Assumption: pose in int32
    min_x = 1e10
    max_x = -1e10
    min_y = 1e10
    max_y = -1e10

    bbox = torch.IntTensor(4)

    for joint in pose:
        x = joint[0].item()
        y = joint[1].item()
        try:
            vis = joint[2].item()
        except:
            vis = 1

        if vis < 0.001:
            continue

        if x < 0 or y < 0:
            continue

        if x < min_x:
            min_x = x

        if x > max_x:
            max_x = x

        if y < min_y:
            min_y = y

        if y > max_y:
            max_y = y

    bbox[0] = min_x
    bbox[1] = min_y
    bbox[2] = max_x
    bbox[3] = max_y

    original_bbox = bbox.clone()

    original_bbox_width = torch.abs(bbox[0] - bbox[2]).item()
    original_bbox_height = torch.abs(bbox[1] - bbox[3]).item()
    original_window_size = torch.IntTensor([max(original_bbox_height, original_bbox_width), max(original_bbox_height, original_bbox_width)])
    original_center = torch.IntTensor([
        original_bbox[2] - original_bbox_width / 2,
        original_bbox[3] - original_bbox_height / 2
    ])

    if bbox_offset is not None:
        half_offset = int(bbox_offset / 2.0)
        bbox[0] = bbox[0] - half_offset
        bbox[1] = bbox[1] - half_offset
        bbox[2] = bbox[2] + half_offset
        bbox[3] = bbox[3] + half_offset

    bbox_width = torch.abs(bbox[0] - bbox[2]).item()
    bbox_height = torch.abs(bbox[1] - bbox[3]).item()

    window_size = torch.IntTensor([max(bbox_height, bbox_width), max(bbox_height, bbox_width)])
    center = torch.IntTensor([
        bbox[2] - bbox_width / 2,
        bbox[3] - bbox_height / 2
    ])

    to_return = {
        "original_bbox": original_bbox,
        "original_center": original_center,
        "original_window_size": original_window_size,
        "offset_bbox": bbox,
        "offset_center": center,
        "offset_window_size": window_size
    }

    return to_return
    