import os
import shutil
import random
import torch

import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from matplotlib.patches import Rectangle
from datasets.MPIIDataset import MPIIDataset, mpii_joint_order
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

subfolders = ["train_norandom_nosaved", "train_random_nosaved", "val_norandom_nosaved", "train_norandom_saved", "train_random_saved", "val_norandom_saved"]
scenarios = [[True, False, False, False], [True, False, True, False], [True, True, False, False], [True, False, False, True], [True, False, True, True], [True, True, False, True]] # train, val, random, saved

for path in subfolders:
    os.makedirs(output_folder + "/" + path)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

for scenario_idx, scenario in enumerate(scenarios):
   
    train = scenario[0]
    val = scenario[1]
    use_random = scenario[2]
    saved = scenario[3]
    
    ds = MPIIDataset("/data/mjakobs/data/mpii/", train=train, val=val, use_random_parameters=use_random, use_saved_tensors=saved)

    all_indices = list(range(len(ds)))
    random.seed(1)
    random.shuffle(all_indices)
    test_indices = all_indices[:2]

    print("train {} val {} random {} saved {}".format(train, val, use_random, saved))

    for idx in test_indices:

        entry = ds[idx]

        parameters = entry["parameters"]
        if val:
            assert parameters[0] == 1.0 # scale
            assert parameters[1] == 0.0 # angle
            assert parameters[2] == 0.0 # flip
            assert parameters[3] == 0.0 # trans x
            assert parameters[4] == 0.0 # trans y

        image = entry["normalized_image"].permute(1, 2, 0)

        assert image.max() <= 1 and image.min() >= -1

        image = (image + 1) / 2.0

        plt.subplot(121)
        plt.imshow(image)

        pose = entry["normalized_pose"]
        vis = pose[:, 2]
        vis_int = vis.long()

        x = (pose[:, 0] * 255.0 * vis).numpy()
        y = (pose[:, 1] * 255.0 * vis).numpy()

        assert pose.max() <= 1 and (pose.min() >= 0 or pose.min() == -1e9), print(entry["normalized_pose"])
        for o in range(16):
            assert vis_int[o] == 1 or vis_int[o] == 0 

        x[x == 0] = None
        y[y == 0] = None

        for i in range(16):
            plt.scatter(x=x[i], y=y[i], label="{}".format(mpii_joint_order[i]))

        fontP = FontProperties()
        fontP.set_size('x-small')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=3, fancybox=True, prop=fontP)

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

        for i in range(16):
            plt.scatter(x=x[i], y=y[i], label="{}".format(mpii_joint_order[i]))

        for i, (src, dst) in enumerate(joint_mapping):
            if not vis[src]:
                continue

            if vis[dst]:
                plt.plot([x[src], x[dst]], [y[src], y[dst]], lw=1, c="#00FFFF")

        bbox = entry["bbox"]
        bbox_rect = Rectangle((bbox[0], bbox[1]), abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3]), linewidth=1, facecolor='none', edgecolor="#FF00FF", clip_on=False)
        ax = plt.gca()
        ax.add_patch(bbox_rect)

        if use_random:
            if saved:
                parameters = entry["parameters"]
                parameter_text = "scale={}, angle={}, flip_lr={}".format(parameters[0], int(parameters[1]), int(parameters[2]))
            else:
                parameter_text = "scale={}, angle={}, flip_lr={}".format(ds.aug_conf["scale"], ds.aug_conf["angle"], ds.aug_conf["flip"])
            plt.figtext(0.15, 0.15, parameter_text, fontsize=14)
            plt.subplots_adjust(bottom=0.20)


        plt.savefig(output_folder + "/" + subfolders[scenario_idx] + "/" + str(idx) + ".png")
        plt.close()
