import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.transform import resize

from datasets import mpii_joint_order

import os

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

def show_predictions_ontop(image, poses, path):
    #heatmaps = heatmaps[-1, :, :, :]
    #single_image = images[-1, :, :, :].reshape(256, 256, 3)
    #single_image = resize(single_image, (128, 128))

    image = image.reshape(256, 256 , 3)

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    
    for i, axi in enumerate(ax.flat):
        axi.set_xticks([])
        axi.set_yticks([])

        pose_coordinates = poses[i].detach().numpy()
        axi.imshow(image)
        axi.scatter(pose_coordinates[0], pose_coordinates[1], c="#FF00FF")
        axi.set_title(mpii_joint_order[i] + " vis: {0:.2f}".format(pose_coordinates[2]))
        #axi.imshow(resize(heatmap, (128, 128)), alpha=0.5)

    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path, pad_inches=0.01)

