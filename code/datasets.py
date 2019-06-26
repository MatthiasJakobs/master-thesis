import torch
import torch.utils.data as data

import os
import re
import glob

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pandas as pd

from skimage import io

def show_pose(clip, idx):
    image = clip["images"][idx]
    pose = clip["poses"][idx]
    visibility = clip["visibility"][idx]

    show_pose_on_image(image, pose, visibility)

def show_pose_on_image(image, pose, visibility):

    plt.figure()
    plt.imshow(image)
    for i,joint in enumerate(pose):
        x = joint[0]
        y = joint[1]
        if visibility[i]:
            c = "g"
        else:
            c = "r"

        plt.scatter(x, y, s=10, marker="*", c=c)

    plt.pause(0.001)
    plt.show()

def show_pose_mpii(image, label):
    xs = label["pose"]["x"]
    ys = label["pose"]["y"]
    vis = label["pose"]["visible"]

    print(label["pose"])

    plt.figure()
    plt.imshow(image)

    for i,joint in enumerate(xs):
        x = xs[i]
        y = ys[i]

        if vis[i]:
            c = "g"
        else:
            c = "r"
        print(x,y)
        plt.scatter(x, y, s=10, marker="*", c=c)

    # print objpose in blue
    print("objpose", label["obj_pose"][0], label["obj_pose"][1])
    plt.scatter(label["obj_pose"][0], label["obj_pose"][1], s=10, marker="*", c="b")
    plt.pause(0.001)
    plt.show()


class PennActionDataset(data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.items = sorted(os.listdir(self.root_dir + "frames"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label_path = self.root_dir + "labels/" + self.items[idx]
        print(label_path)
        label = sio.loadmat(label_path)
        images = []
        frame_folder = self.root_dir + "frames/" + self.items[idx] + "/"
        print(frame_folder)
        all_images = sorted(os.listdir(frame_folder))
        print(all_images)
        
        pose = []
        for i in range(len(all_images)):
            image = io.imread(frame_folder + all_images[i])
            images.append(image)

            joint_frame = []
            for o in range(13):
                joint_coordinate = [label["x"][i][o], label["y"][i][o]]
                joint_frame.append(joint_coordinate)

            pose.append(joint_frame)

        action = label["action"][0]
        visibility = label["visibility"]

        return {"action": action, "images": np.array(images), "poses": np.array(pose), "visibility": np.array(visibility)}


def superflatten(array):
    return array.flatten()[0]

class MPIIDataset(data.Dataset):

    '''
    What is (probably) happening with the train / val split of luvizon:
        - They load training data only and extract each person. Afterwards, they split it up
          into train / val and save it.
        - When I do it like below, almost exactly size(me) = size(train_luvizon) + size(val_luvizon)
    '''

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        assert "annotations.mat" in os.listdir(self.root_dir)

        annotations = sio.loadmat(self.root_dir + "annotations")["RELEASE"]
        
        train_binary = annotations["img_train"][0][0][0]
        train_indeces = np.where(np.array(train_binary))[0]
        
        self.labels = []
        missing_annnotation_count = 0

        for idx in train_indeces:
            label = annotations["annolist"][0][0][0][idx]
            image_name = label["image"]["name"][0][0][0]

            annorect = label["annorect"]
            if len(annorect) == 0:
                # some labels are not present in the annotation file
                missing_annnotation_count += 1
                continue


            for rect_id in range(len(annorect[0])):
                ann = annorect[0][rect_id]
                head_coordinates = [ 
                    superflatten(ann["x1"]), 
                    superflatten(ann["y1"]), 
                    superflatten(ann["x2"]), 
                    superflatten(ann["y2"])
                ] # rect x1, y1, x2, y2
                try:
                    scale = superflatten(ann["scale"])
                    
                    obj_pose = [
                        # rough position of human body (x,y)
                        superflatten(superflatten(ann["objpos"]["x"])),
                        superflatten(superflatten(ann["objpos"]["y"]))
                    ]

                    point = superflatten(ann["annopoints"]["point"])
                    
                    xs = list(map(lambda x: superflatten(x), point["x"].flatten()))
                    ys = list(map(lambda x: superflatten(x), point["y"].flatten()))

                    # 0:r ankle 1:r knee 2:r hip 3:l hip 4:l knee 5:l ankle 6:pelvis 7:thorax 8:upper neck 9:head top 10:r wrist 11:r elbow 12:r shoulder 13:l shoulder 14:l elbow 15:l wrist
                    ids = list(map(lambda x: superflatten(x), point["id"].flatten()))
                    vs = []
                    for i,v in enumerate(point["is_visible"].flatten()):
                        try:
                            if ids[i] == 8 or ids[i] == 9:
                                # for some reason, upper neck and head annotations are always empty, probably  because they are always visible(?)
                                # set them to be always visible
                                vs.append(1)
                            else:
                                vs.append(superflatten(v))
                        except IndexError:
                            vs.append(0)
                except (IndexError, ValueError):
                    # all these fields are necessary, thus: skip if not present
                    missing_annnotation_count = missing_annnotation_count + 1
                    continue

                pose = {"x": xs, "y": ys, "visible": vs, "ids": ids}

                final_label = {
                    "head": head_coordinates,
                    "scale": scale,
                    "obj_pose": obj_pose,
                    "pose": pose,
                    "image_name": image_name
                }

                self.labels.append(final_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        full_image_path = self.root_dir + "images/" + label["image_name"]
        image = io.imread(full_image_path)

        return image, label

    
class JHMDBDataset(data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.items = sorted(glob.glob(self.root_dir + "*/*"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_path = self.items[idx]
        relative_path_split = item_path[len(self.root_dir):].split("/")
        action = relative_path_split[0]
        
        label = sio.loadmat(item_path + "/joint_positions")
        pose = label["pos_img"].T
        
        all_images = glob.glob(item_path + "/*.png")
        images = []
        for image_path in all_images:
            images.append(io.imread(image_path))

        image_height = len(images[0])
        image_width = len(images[0][0])

        visibility = []
        for frame_pose in pose:
            frame_visibility = []
            for joint in frame_pose:
                x = joint[0]
                y = joint[1]
                
                if x < 0 or x > image_width or y < 0 or y > image_height:
                    frame_visibility.append(0)
                else:
                    frame_visibility.append(1)
            visibility.append(frame_visibility)

        return {"action": action, "poses": pose, "images": images, "visibility": visibility}


#based on https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
class ImbalancedDatasetSampler(data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx]["class"]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


#train_loader = data.DataLoader(
#    dataset,
#    sampler=ImbalancedDatasetSampler(dataset, num_samples=10),
#    batch_size=5
#)

#for image in train_loader:
#    print(image)
