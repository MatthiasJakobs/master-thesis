import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.transform import resize

from datasets import mpii_joint_order

import os

import torch

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

def show_predictions_ontop(ground_truth, image, poses, path, matrix, original_size):

    plt.xticks([])
    plt.yticks([])
    
    ground_pose = ground_truth[:, 0:2]
    predicted_pose = poses[:, 0:2]

    plt.imshow(image)

    orig_gt_coordinates = transform_pose(matrix, ground_pose, inverse=True)
    orig_pred_coordinates = transform_pose(matrix, predicted_pose, inverse=True)

    gt_color = "r"
    pred_color = "b"

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

    for i, (src, dst) in enumerate(mapping):
        if not ground_truth[src, 2]:
            continue

        if ground_truth[dst, 2]:
            #print("{} => {}".format(mpii_joint_order[src], mpii_joint_order[dst]))
            plt.plot([orig_gt_coordinates[src][0], orig_gt_coordinates[dst][0]], [orig_gt_coordinates[src][1], orig_gt_coordinates[dst][1]], lw=1, c=gt_color)
            plt.plot([orig_pred_coordinates[src][0], orig_pred_coordinates[dst][0]], [orig_pred_coordinates[src][1], orig_pred_coordinates[dst][1]], lw=1, c=pred_color)

            plt.scatter([orig_gt_coordinates[src][0], orig_gt_coordinates[dst][0]], [orig_gt_coordinates[src][1], orig_gt_coordinates[dst][1]], c=gt_color)
            plt.scatter([orig_pred_coordinates[src][0], orig_pred_coordinates[dst][0]], [orig_pred_coordinates[src][1], orig_pred_coordinates[dst][1]], c=pred_color)
        else:
            #print("{}".format(mpii_joint_order[src]))
            plt.scatter(orig_gt_coordinates[src][0], orig_gt_coordinates[src][1], c=gt_color)
            plt.scatter(orig_pred_coordinates[src][0], orig_pred_coordinates[src][1], c=pred_color)

        


    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)
    plt.close()
    #plt.pause(0.001)
    #plt.show()


