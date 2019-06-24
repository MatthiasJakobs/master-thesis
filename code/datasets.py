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

        for i, idx in enumerate(train_indeces):
            label = annotations["annolist"][0][0][0][idx]
            image_name = label["image"]["name"][0][0][0]

            annorect = label["annorect"]
            if len(annorect) == 0:
                # some labels are not present in the annotation file
                missing_annnotation_count += 1
                continue


            for rect_id in range(len(annorect[0])):
                ann = annorect[0][rect_id]
                
                if len(ann) < 7 or len(ann) == 22 or len(ann[4]) == 0:
                    missing_annnotation_count += 1
                    continue
                
                head_coordinates = [ ann[0][0][0], ann[1][0][0], ann[2][0][0], ann[3][0][0]] # rect x1, y1, x2, y2

                # some annotations contain additional fields, thus need to compensate indice choice
                scale_index = len(ann)-2
                obj_pose_index = len(ann)-1
        
                obj_pose = [ann[obj_pose_index][0][0][0][0][0], ann[obj_pose_index][0][0][1][0][0]] # rough position of human body (x,y)

                scale = ann[scale_index][0][0]

                pose = {"x": [], "y": [], "visible": [], "ids": []}

                for joint_label in ann[4][0][0][0][0]:
                    pose["x"].append(joint_label[0][0][0])
                    pose["y"].append(joint_label[1][0][0])
                    pose["ids"].append(joint_label[2][0][0])
                    
                    try:
                        visible = joint_label[3][0][0]
                    except IndexError:
                        # For some reason, sometimes "not visible" is encoded as 0 and sometimes as an empty array
                        visible = 0
                    
                    pose["visible"].append(visible)

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
