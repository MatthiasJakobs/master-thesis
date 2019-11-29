from datasets.MPIIDataset import MPIIDataset
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import numpy as np
from deephar.utils import transform_pose
joint_mapping = [
    [0, 1],
    [1, 2],
    [2, 6],
    [3, 6],
    [4, 3],
    [5, 4],
    [6, 7],
    [7, 8],
    [8, 9],
    [10, 11],
    [11, 12],
    [12, 7],
    [13, 7],
    [14, 13],
    [15, 14]
]

def plot_mpii_augmented():
    ds = MPIIDataset("/data/mjakobs/data/mpii/", train=True, val=False, use_random_parameters=False, use_saved_tensors=False)
    ds_augmented = MPIIDataset("/data/mjakobs/data/mpii/", train=True, val=False, use_random_parameters=True, use_saved_tensors=False)

    ds_augmented.angles=np.array([-15])
    ds_augmented.scales=np.array([1.3])
    ds_augmented.flip_horizontal = np.array([1])

    entry = ds[20]
    image = entry["normalized_image"]
    image = image.permute(1,2,0)
    image = (image + 1) / 2.0

    plt.subplot(1,4,3)
    plt.imshow(image)

    pose = entry["normalized_pose"]
    scaled_pose = pose * 255.0

    for i, (src, dst) in enumerate(joint_mapping):
        plt.plot([scaled_pose[src][0], scaled_pose[dst][0]], [scaled_pose[src][1], scaled_pose[dst][1]], lw=1, c="#00FFFF")
        plt.scatter(scaled_pose[src][0], scaled_pose[src][1], s=10, c="#FF00FF")
        plt.scatter(scaled_pose[dst][0], scaled_pose[dst][1], s=10, c="#FF00FF")

    plt.xticks([])
    plt.yticks([])

    image_number = "{}".format(int(entry["image_path"][0].item()))
    image_name = "{}.jpg".format(image_number.zfill(9))
    image = io.imread("/data/mjakobs/data/mpii/images/{}".format(image_name))
    plt.subplot(1,4,1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,4,2)
    plt.imshow(image)
    matrix = entry["trans_matrix"]
    pose = transform_pose(matrix, entry["normalized_pose"][:, 0:2], inverse=True)
    scaled_pose = pose

    for i, (src, dst) in enumerate(joint_mapping):
        plt.plot([scaled_pose[src][0], scaled_pose[dst][0]], [scaled_pose[src][1], scaled_pose[dst][1]], lw=1, c="#00FFFF")
        plt.scatter(scaled_pose[src][0], scaled_pose[src][1], s=10, c="#FF00FF")
        plt.scatter(scaled_pose[dst][0], scaled_pose[dst][1], s=10, c="#FF00FF")
    plt.xticks([])
    plt.yticks([])

    entry = ds_augmented[20]
    image = entry["normalized_image"]
    image = image.permute(1,2,0)
    image = (image + 1) / 2.0

    plt.subplot(1,4,4)
    plt.imshow(image)
    pose = entry["normalized_pose"]
    scaled_pose = pose * 255.0

    for i, (src, dst) in enumerate(joint_mapping):
        plt.plot([scaled_pose[src][0], scaled_pose[dst][0]], [scaled_pose[src][1], scaled_pose[dst][1]], lw=1, c="#00FFFF")
        plt.scatter(scaled_pose[src][0], scaled_pose[src][1], s=10, c="#FF00FF")
        plt.scatter(scaled_pose[dst][0], scaled_pose[dst][1], s=10, c="#FF00FF")

    plt.xticks([])
    plt.yticks([])

    plt.show()

def plot_example_mpii():
    ds = MPIIDataset("/data/mjakobs/data/mpii/", train=True, val=False, use_random_parameters=False, use_saved_tensors=False)
    plot_index = 1
    for i in range(102, 502, 100):
        entry = ds[i]
        image_number = "{}".format(int(entry["image_path"][0].item()))
        image_name = "{}.jpg".format(image_number.zfill(9))
        image = io.imread("/data/mjakobs/data/mpii/images/{}".format(image_name))
        plt.subplot(2,2,plot_index)
        plot_index += 1
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

    plt.show()