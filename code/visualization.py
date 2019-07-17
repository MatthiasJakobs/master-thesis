import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.transform import resize

from datasets import mpii_joint_order

import os

import torch

from deephar.utils import transform_2d_point

def show_pose(clip, idx):
    image = clip["images"][idx]
    pose = clip["poses"][idx]

    show_pose_on_image(image, pose)

def show_pose_on_image(image, pose):

    plt.figure()
    plt.imshow(image)
    for i,joint in enumerate(pose):
        x = joint[0]
        y = joint[1]
        visible = joint[2]

        if visible:
            plt.scatter(x, y, s=10, marker="*", c="g")

    plt.pause(0.001)
    plt.show()

def show_pose_mpii(annotation):
    xs = annotation["original_pose"][:,0]
    ys = annotation["original_pose"][:,1]
    vis = annotation["original_pose"][:,2]

    plt.figure()
    plt.imshow(annotation["original_image"])

    for i,joint in enumerate(xs):
        x = xs[i]
        y = ys[i]

        if vis[i]:
            c = "g"
        else:
            c = "r"

        plt.scatter(x, y, s=10, marker="*", c=c)

    # print objpose in blue
    if "center" in annotation.keys():
        plt.scatter(annotation["center"][0], annotation["center"][1], s=10, marker="*", c="#FF00FF")

    # print bounding box if present
    # if annotation["bbox"]:
    #     bbox = annotation["bbox"]
    #     ax = plt.gca()
    #     # lower left: (x1, y2)

    #     start_point = (bbox[0],bbox[1])

    #     width = abs(bbox[0] - bbox[2])
    #     height = abs(bbox[1] - bbox[3])
    #     print(width, height)
    #     #rect = Rectangle(start_point,width,height,linewidth=1,edgecolor='r',facecolor='none')
    #     rect = Rectangle(start_point, width, height, linewidth=1,edgecolor='r',facecolor='none')
    #     ax.add_patch(rect)

    plt.pause(0.001)
    plt.show()

def show_predictions_ontop(ground_truth, image, poses, path, matrix, original_size):
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    for i, axi in enumerate(ax.flat):
        axi.set_xticks([])
        axi.set_yticks([])

        if torch.cuda.is_available():
            pose_coordinates = poses[i].cpu().detach().numpy()
            gt_coordintates = ground_truth[i].cpu().detach().numpy()
        else:
            pose_coordinates = poses[i].detach().numpy()
            gt_coordintates = ground_truth[i].detach().numpy()

        axi.imshow(image)

        vis = pose_coordinates[2]
        if ground_truth[i, 2] == 0.0:
            axi.set_title(mpii_joint_order[i] + " not visible")
            continue
        else:
            axi.set_title(mpii_joint_order[i] + " vis: {0:.2f}".format(vis))

        gt_coordintates = transform_2d_point(matrix, gt_coordintates[0:2], inverse=True)
        pose_coordinates = transform_2d_point(matrix, pose_coordinates[0:2], inverse=True)

        axi.scatter(gt_coordintates[0], gt_coordintates[1], c="b")
        axi.scatter(pose_coordinates[0], pose_coordinates[1], c="#FF00FF")

    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path, pad_inches=0.01)
    plt.close()

