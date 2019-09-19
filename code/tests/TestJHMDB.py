import os
import shutil
import random
import torch
import glob

import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from datasets.JHMDBDataset import JHMDBDataset, actions
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

output_folder = "/tmp/TestJHMDB"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

subfolders = ["train_norandom_nosaved", "train_random_nosaved", "test_norandom_nosaved", "test_random_nosaved"]
scenarios = [[True, False, False], [True, True, False], [False, False, False], [False, True, False]]

for path in subfolders:
    os.makedirs(output_folder + "/" + path)


for scenario_idx, scenario in enumerate(scenarios):
    train = scenario[0]
    use_random_parameters = scenario[1]
    use_saved_tensors = scenario[2]
    
    ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=train, use_random_parameters=use_random_parameters, use_saved_tensors=False)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)
    test_indices = all_indices[:10]

    for idx in test_indices:

        entry = ds[idx]

        for frame in range(entry["sequence_length"].item()):

            image = entry["normalized_frames"][frame].reshape(255, 255, 3)
            action_label = actions[entry["action_1h"].argmax().item()]

            assert image.max() <= 1 and image.min() >= -1

            image = (image + 1) / 2.0

            plt.subplot(121)
            plt.imshow(image)

            pose = entry["normalized_poses"][frame]
            vis = pose[:, 2]
            x = (pose[:, 0] * 255.0 * vis).numpy()
            y = (pose[:, 1] * 255.0 * vis).numpy()

            assert pose.max() <= 1 and pose.min() >= 0
            for o in range(16):
                vis_int = vis.numpy().astype(np.int)
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

            matrix = entry["trans_matrices"][frame]
            index = int(entry["index"].item())
            
            item_path = ds.items[index]
            all_frames = sorted(glob.glob(item_path + "/*.png"))

            image_path = all_frames[frame]
            image = io.imread(image_path)
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

            plt.figtext(0.40, 0.03, action_label, fontsize=16)

            if use_random_parameters:
                parameter_text = "scale={}, angle={}, flip_lr={}, trans_x={}, trans_y={}".format(ds.test["scale"], ds.test["angle"], ds.test["flip"], ds.test["trans_x"], ds.test["trans_y"])
                plt.figtext(0.07, 0.15, parameter_text, fontsize=14)
                plt.subplots_adjust(bottom=0.30)
            else:
                plt.subplots_adjust(bottom=0.15)



            plt.savefig(output_folder + "/" + subfolders[scenario_idx] + "/" + str(idx) + "_" + str(frame) + ".png")
            plt.close()
