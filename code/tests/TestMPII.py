import os
import shutil
import random
import torch

import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from datasets.MPIIDataset import MPIIDataset
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

output_folder = "/tmp/TestMPII"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

subfolders = ["norandom_nosaved", "random_nosaved", "norandom_saved", "random_saved"]
scenarios = [[False, False], [True, False], [False, True], [True, True]]

for path in subfolders:
    os.makedirs(output_folder + "/" + path)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


ds = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=False, use_saved_tensors=False)

all_indices = list(range(len(ds)))
random.shuffle(all_indices)
test_indices = all_indices[:20]

for scenario_idx, scenario in enumerate(scenarios):
    use_random_parameters = scenario[0]
    use_saved_tensors = scenario[1]
    
    ds = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=use_random_parameters, use_saved_tensors=False)

    for idx in test_indices:

        entry = ds[idx]

        image = entry["normalized_image"].reshape(256, 256, 3)

        assert image.max() <= 1 and image.min() >= -1

        image = (image + 1) / 2.0

        plt.subplot(121)
        plt.imshow(image)

        pose = entry["normalized_pose"]
        vis = pose[:, 2]
        vis_int = vis.long()

        x = (pose[:, 0] * 256.0 * vis).numpy()
        y = (pose[:, 1] * 256.0 * vis).numpy()

        assert pose.max() <= 1 and (pose.min() >= 0 or pose.min() == -1e9), print(entry["normalized_pose"])
        for o in range(16):
            assert vis_int[o] == 1 or vis_int[o] == 0 

        x[x == 0] = None
        y[y == 0] = None

        plt.scatter(x=x, y=y, c="#FF00FF")

        for i, (src, dst) in enumerate(joint_mapping):
            if not vis[src]:
                continue

            if vis[dst]:
                plt.plot([x[src], x[dst]], [y[src], y[dst]], lw=1, c="#00FFFF")


        plt.subplot(122)

        matrix = entry["trans_matrix"]

        image_number = str(entry["image_path"].item()).zfill(9)
        image = io.imread("/data/mjakobs/data/mpii/images/{}.jpg".format(image_number))
        plt.imshow(image)

        pose = torch.from_numpy(transform_pose(matrix, pose[:, 0:2], inverse=True)).float()
        x = (pose[:, 0] * vis).numpy()
        y = (pose[:, 1] * vis).numpy()

        x[x == 0] = None
        y[y == 0] = None

        plt.scatter(x=x, y=y, c="#FF00FF", s=10)

        for i, (src, dst) in enumerate(joint_mapping):
            if not vis[src]:
                continue

            if vis[dst]:
                plt.plot([x[src], x[dst]], [y[src], y[dst]], lw=1, c="#00FFFF")

        bbox = entry["bbox"]
        bbox_rect = Rectangle((bbox[0], bbox[1]), abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3]), linewidth=1, facecolor='none', edgecolor="#FF00FF", clip_on=False)
        ax = plt.gca()
        ax.add_patch(bbox_rect)

        if use_random_parameters:
            if use_saved_tensors:
                parameters = entry["parameters"]
                parameter_text = "scale={}, angle={}, flip_lr={}".format(parameters[0], int(parameters[1]), int(parameters[2]))
            else:
                parameter_text = "scale={}, angle={}, flip_lr={}".format(ds.test["scale"], ds.test["angle"], ds.test["flip"])
            plt.figtext(0.15, 0.15, parameter_text, fontsize=14)
            plt.subplots_adjust(bottom=0.20)


        plt.savefig(output_folder + "/" + subfolders[scenario_idx] + "/" + str(idx) + ".png")
        plt.close()
