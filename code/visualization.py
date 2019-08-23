import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.transform import resize

from datasets import mpii_joint_order
from collections import OrderedDict

import os

import torch
import numpy as np

from deephar.utils import transform_2d_point, transform_pose

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

def show_predictions_ontop(ground_truth, image, poses, path, matrix, bbox=None, save=True):

    plt.xticks([])
    plt.yticks([])

    ground_pose = ground_truth[:, 0:2]
    predicted_pose = poses[:, 0:2]

    plt.imshow(image)

    orig_gt_coordinates = transform_pose(matrix, ground_pose, inverse=True)
    orig_pred_coordinates = transform_pose(matrix, predicted_pose, inverse=True)

    gt_color = "r"
    pred_color = "b"
    bbox_color = "#ff00ff"

    mapping = [
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

    colors = [
        "#1176a5",
        "#fdfe5f",
        "#abb633",
        "#17af4f",
        "#8d403c",
        "#b7c0d3",
        "#c0922c",
        "#486054",
        "#d4696f",
        "#2fc64f",
        "#d53606",
        "#52043f",
        "#14a796",
        "#831b4c",
        "#0d0072",
        "#f65fb1"
    ]
    for i, (src, dst) in enumerate(mapping):
        if not ground_truth[src, 2]:
            continue

        if ground_truth[dst, 2]:
            #print("{} => {}".format(mpii_joint_order[src], mpii_joint_order[dst]))
            plt.plot([orig_gt_coordinates[src][0], orig_gt_coordinates[dst][0]], [orig_gt_coordinates[src][1], orig_gt_coordinates[dst][1]], lw=1, c=gt_color)
            plt.plot([orig_pred_coordinates[src][0], orig_pred_coordinates[dst][0]], [orig_pred_coordinates[src][1], orig_pred_coordinates[dst][1]], lw=1, c=pred_color)
            plt.scatter(orig_gt_coordinates[src][0], orig_gt_coordinates[src][1], label="{}".format(mpii_joint_order[src]), c=colors[src])
            plt.scatter(orig_pred_coordinates[src][0], orig_pred_coordinates[src][1], c=colors[src])
            
            plt.scatter(orig_gt_coordinates[dst][0], orig_gt_coordinates[dst][1], label="{}".format(mpii_joint_order[dst]), c=colors[dst])
            plt.scatter(orig_pred_coordinates[dst][0], orig_pred_coordinates[dst][1], c=colors[dst])
        else:
            #print("{}".format(mpii_joint_order[src]))
            plt.scatter(orig_gt_coordinates[src][0], orig_gt_coordinates[src][1], label="{}".format(mpii_joint_order[src]), c=colors[src])
            plt.scatter(orig_pred_coordinates[src][0], orig_pred_coordinates[src][1], c=colors[src])

    if bbox is not None:
        bbox_rect_original = Rectangle((bbox[0], bbox[1]), abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3]), linewidth=1, facecolor='none', edgecolor=bbox_color)
        ax = plt.gca()
        ax.add_patch(bbox_rect_original)

    # remove duplicate handles in legend (src: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if save:
        if os.path.isfile(path):
            os.remove(path)
        plt.savefig(path)
    else:
        plt.show()

    plt.close()

    #plt.pause(0.001)
    #plt.show()


