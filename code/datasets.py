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

def create_custom_mpii(root_dir):
    print("create custom mpii into {}".format(root_dir))

    labels = sio.loadmat(root_dir + "mpii_human_pose_v1_u12_1")["RELEASE"]
    total_length = len(labels["annolist"][0][0][0])

    # at this moment, I only use training data (for testing) because they have full annotation
    train_binary = labels["img_train"][0][0][0]
    indexes = np.where(np.array(train_binary))[0]
    
    data = []

    for i, idx in enumerate(indexes):
        label = labels["annolist"][0][0][0][idx]
        image_name = label["image"]["name"][0][0][0]

        #if len(labels["act"][0][0][idx][0][1]) == 0:
        #    continue

        # multiple action annotations possible
        #actions = list(map(lambda x: x.strip(), labels["act"][0][0][idx][0][1][0].split(',')))

        # TODO: 
        # MPII has an attribute suggesting "sufficiently seperated persons".
        # Here, I chose to use the first in the list (if any) and "1" (the first person in the image) else
        # Need to check with code how they used it.
        # For now, this should work fine for testing

        try:
            single_person_id = labels["single_person"][0][0][idx][0][0][0]
        except IndexError:
            single_person_id = 1

        if len(label["annorect"]) == 0:
            continue

        try:
            if len(label["annorect"]["annopoints"][0][0]) == 0:
                continue
        except ValueError:
            continue

        try:
            x_coords = list(map(lambda x: x[0][0], label["annorect"]["annopoints"][0][single_person_id - 1]["point"][0][0]["x"][0]))
            y_coords = list(map(lambda x: x[0][0], label["annorect"]["annopoints"][0][single_person_id - 1]["point"][0][0]["y"][0]))
            visibility = list(map(lambda x: x[0][0] if len(x) > 0 else 0, label["annorect"]["annopoints"][0][single_person_id - 1]["point"][0][0]["is_visible"][0]))
        except ValueError:
            print("no coordinates for index {}".format(idx))
        except Exception as error:
            print(error)
            print("error for index {}".format(idx))
            return

        pose = np.array([x_coords, y_coords]).T

        frame_data = [i, image_name]
        frame_data.extend([item for sublist in pose for item in sublist])
        frame_data.extend(visibility)

        data.append(frame_data)
        

    df = pd.DataFrame(data, columns=["id", "image_name", "joint_x_1", "joint_y_1", "joint_x_2", "joint_y_2", "joint_x_3", "joint_y_3", "joint_x_4", "joint_y_4", "joint_x_5", "joint_y_5", "joint_x_6","joint_y_6", "joint_x_7", "joint_y_7", "joint_x_8", "joint_y_8", "joint_x_9", "joint_y_9","joint_x_10", "joint_y_10", "joint_x_11", "joint_y_11", "joint_x_12", "joint_y_12", "joint_x_13", "joint_y_13", "joint_x_14", "joint_y_14","joint_x_15", "joint_y_15", "joint_x_16","joint_y_16", "visibility_1", "visibility_2", "visibility_3", "visibility_4", "visibility_5", "visibility_6", "visibility_7", "visibility_8", "visibility_9", "visibility_10", "visibility_11", "visibility_12", "visibility_13", "visibility_14", "visibility_15", "visibility_16"])
    df.to_csv(root_dir + "custom_mpii.csv")


class MPIIDataset(data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        if not "custom_mpii.csv" in os.listdir(self.root_dir):
            create_custom_mpii(self.root_dir)
        
        self.labels = pd.read_csv(self.root_dir + "custom_mpii.csv")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels.iloc[idx].to_dict()

        full_image_path = self.root_dir + "images/" + item["image_name"]
        image = io.imread(full_image_path)

        pose = []
        visibility = []
        for i in range(1,17):
            x = item["joint_x_" + str(i)]
            y = item["joint_y_" + str(i)]
            v = item["visibility_" + str(i)]

            pose.append([x,y])
            visibility.append(v)
        
        return {"pose": pose, "image": image, "visibility": visibility}

    
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
