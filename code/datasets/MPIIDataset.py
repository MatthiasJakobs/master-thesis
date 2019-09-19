import torch
import torch.utils.data as data

import random

import os
import re
import glob

import scipy.io as sio
import numpy as np
import pandas as pd

from skimage import io
from skimage.transform import resize

from deephar.image_processing import center_crop, rotate_and_crop, normalize_channels
from deephar.utils import transform_2d_point, translate, scale, flip_h, superflatten, transform_pose, get_valid_joints

class MPIIDataset(data.Dataset):

    '''
    What is (probably) happening with the train / val split of luvizon:
        - They load training data only and extract each person. Afterwards, they split it up
          into train / val and save it.
        - When I do it like below, almost exactly size(me) = size(train_luvizon) + size(val_luvizon)
    '''

    def __init__(self, root_dir, transform=None, use_random_parameters=True, use_saved_tensors=False):

        self.root_dir = root_dir

        assert "annotations.mat" in os.listdir(self.root_dir)

        annotations = sio.loadmat(self.root_dir + "annotations")["RELEASE"]

        train_binary = annotations["img_train"][0][0][0]
        train_indeces = np.where(np.array(train_binary))[0]

        self.final_size=256

        self.use_saved_tensors = use_saved_tensors
        self.use_random_parameters = use_random_parameters

        if self.use_random_parameters:
            self.prefix = "rand_"
        else:
            self.prefix = ""

        if not os.path.exists(root_dir + self.prefix + "tensors"):
            os.makedirs(root_dir + self.prefix + "tensors")

        if use_random_parameters:
            self.angles=np.array(range(-40, 40+1, 5))
            self.scales=np.array([0.7, 1., 1.3])
            self.channel_power_exponent = 0.01*np.array(range(90, 110+1, 2))
            self.flip_horizontal = np.array([0, 1])
        else:
            self.angles=np.array([0, 0, 0])
            self.scales=np.array([1., 1., 1.])
            self.flip_horizontal = np.array([0, 0])
            self.channel_power_exponent = None

        self.labels = []
        missing_annnotation_count = 0

        for idx in train_indeces:
            label = annotations["annolist"][0][0][0][idx]
            image_name = label["image"]["name"][0][0][0]

            full_image_path = self.root_dir + "images/{}".format(image_name)
            if not os.path.exists(full_image_path):
                print(full_image_path, "does not exist, skip")
                continue

            annorect = label["annorect"]
            if len(annorect) == 0:
                # some labels are not present in the annotation file
                missing_annnotation_count += 1
                continue


            for rect_id in range(len(annorect[0])):
                ann = annorect[0][rect_id]
                head_coordinates = [
                    float(superflatten(ann["x1"])),
                    float(superflatten(ann["y1"])),
                    float(superflatten(ann["x2"])),
                    float(superflatten(ann["y2"]))
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
                    "head": np.array(head_coordinates),
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
        output = {}

        if self.use_saved_tensors:
            name_path = self.root_dir + "{}tensors/{}".format(self.prefix, label["image_name"])
            t_filepath = torch.load(name_path + ".image_path.pt")
            t_normalized_image = torch.load(name_path + ".normalized_image.pt")
            t_normalized_pose = torch.load(name_path + ".normalized_pose.pt")
            t_original_pose = torch.load(name_path + ".original_pose.pt")
            t_headsize = torch.load(name_path + ".headsize.pt")
            t_trans_matrix = torch.load(name_path + ".trans_matrix.pt")
            t_original_size = torch.load(name_path + ".original_size.pt")
            t_bbox = torch.load(name_path + ".bbox.pt")
            t_parameters = torch.load(name_path + ".parameters.pt")

            output["image_path"] = t_filepath
            output["normalized_image"] = t_normalized_image
            output["normalized_pose"] = t_normalized_pose
            output["original_pose"] = t_original_pose
            output["head_size"] = t_headsize
            output["trans_matrix"] = t_trans_matrix
            output["original_size"] = t_original_size
            output["bbox"] = t_bbox
            output["parameters"] = t_parameters
            

            return output

        full_image_path = self.root_dir + "images/" + label["image_name"]
        image = io.imread(full_image_path)
        original_image = image.copy()

        conf_scale = self.scales[np.random.randint(0, len(self.scales))]
        conf_angle = self.angles[np.random.randint(0, len(self.angles))]
        conf_flip = self.flip_horizontal[np.random.randint(0, len(self.flip_horizontal))]
        if self.channel_power_exponent is not None:
            conf_exponents = np.array([
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))],
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))],
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))]
            ])
        else:
            conf_exponents = None

        self.test = {}
        self.test["scale"] = conf_scale
        self.test["flip"] = conf_flip
        self.test["angle"] = conf_angle

        new_scale = label["scale"] * 1.25 # magic value
        new_objpose = np.array([label["obj_pose"][0], label["obj_pose"][1] + 12 * new_scale]) # magic values, no idea where they are comming from

        window_size = new_scale * conf_scale * 200

        image_width = image.shape[1]
        image_height = image.shape[0]

        bbox = np.array([
            new_objpose[0] - window_size / 2, # x1, upper left
            new_objpose[1] - window_size / 2, # y1, upper left
            new_objpose[0] + window_size / 2, # x2, lower right
            new_objpose[1] + window_size / 2  # y2, lower right
        ])

        # rotate, then crop
        trans_matrix, image = rotate_and_crop(image, conf_angle, new_objpose, (window_size, window_size))
        size_after_rotate = np.array([image.shape[1], image.shape[0]])

        image = resize(image, (self.final_size, self.final_size), preserve_range=True)
        trans_matrix = scale(trans_matrix, self.final_size / size_after_rotate[0], self.final_size / size_after_rotate[1])

        old_pose = np.array([label["pose"]["x"], label["pose"]["y"]]).T
        old_objpos = np.array(label["obj_pose"])

        # randomly flip horizontal
        if conf_flip:
            image = np.fliplr(image)

            trans_matrix = translate(trans_matrix, -image.shape[1] / 2, -image.shape[0] / 2)
            trans_matrix = flip_h(trans_matrix)
            trans_matrix = translate(trans_matrix, image.shape[1] / 2, image.shape[0] / 2)

        trans_matrix = scale(trans_matrix, 1.0 / self.final_size, 1.0 / self.final_size)

        output["center"] = transform_2d_point(trans_matrix, old_objpos)
        t_original_size = torch.tensor([image_height, image_width], requires_grad=False)

        new_x = []
        new_y = []
        for idx, (x, y) in enumerate(old_pose):
            transformed_point = transform_2d_point(trans_matrix, np.array([x,y]))
            new_x.append(transformed_point[0])
            new_y.append(transformed_point[1])

        original_pose = np.empty((16, 3))
        original_pose[:] = np.nan

        for it, joint_index in enumerate(label["pose"]["ids"]):
            x = new_x[it]
            y = new_y[it]

            if x < 0 or y < 0 or x > 1 or y > 1:
                original_pose[joint_index, 0] = np.nan
                original_pose[joint_index, 1] = np.nan
            else:
                original_pose[joint_index, 0] = x
                original_pose[joint_index, 1] = y

            original_pose[joint_index, 2] = label["pose"]["visible"][it]

        original_pose[np.isnan(original_pose)] = -1e9

        normalized_pose = original_pose.copy()

        # According to paper:
        lower_one = np.apply_along_axis(np.all, 1, normalized_pose[:,0:2] < 1.0)
        bigger_zero = np.apply_along_axis(np.all, 1, normalized_pose[:,0:2] > 0.0)

        in_interval = np.logical_and(lower_one, bigger_zero)
        original_pose[:,2] = in_interval
        normalized_pose[:,2] = in_interval

        # calculating head size for pckh (according to paper)
        head_point_upper = np.array([label["head"][0], label["head"][1]])
        head_point_lower = np.array([label["head"][2], label["head"][3]])
        head_size = 0.6 * np.linalg.norm(head_point_upper - head_point_lower)

        image_normalized = normalize_channels(image, power_factors=conf_exponents)

        t_original_image = torch.from_numpy(original_image).float()
        t_bbox = torch.from_numpy(bbox).float()
        t_normalized_image = torch.from_numpy(image_normalized.reshape(3, 256, 256)).float()
        t_normalized_pose = torch.from_numpy(normalized_pose).float()
        t_original_pose = torch.from_numpy(original_pose).float()
        t_headsize = torch.from_numpy(np.array([head_size])).float()
        t_trans_matrix = torch.from_numpy(trans_matrix.copy()).float()

        t_parameters = torch.zeros(3).float()
        t_parameters[0] = self.test["scale"]
        t_parameters[1] = float(self.test["angle"])
        t_parameters[2] = float(self.test["flip"])

        image_number = int(full_image_path[-13:-4])
        
        image_number_np = np.array([image_number])
        t_filepath = torch.from_numpy(image_number_np).int()

        output["bbox"] = t_bbox
        output["image_path"] = t_filepath
        output["normalized_image"] = t_normalized_image
        output["normalized_pose"] = t_normalized_pose
        output["original_pose"] = t_original_pose
        output["head_size"] = t_headsize
        output["trans_matrix"] = t_trans_matrix
        output["original_size"] = t_original_size

        if not self.use_saved_tensors:
            name_path = self.root_dir + "{}tensors/{}".format(self.prefix, label["image_name"])
            if not os.path.exists(name_path + ".image_path.pt"):
                torch.save(t_filepath, name_path + ".image_path.pt")
            if not os.path.exists(name_path + ".normalized_image.pt"):
                torch.save(t_normalized_image, name_path + ".normalized_image.pt")
            if not os.path.exists(name_path + ".normalized_pose.pt"):
                torch.save(t_normalized_pose, name_path + ".normalized_pose.pt")
            if not os.path.exists(name_path + ".original_pose.pt"):
                torch.save(t_original_pose, name_path + ".original_pose.pt")
            if not os.path.exists(name_path + ".headsize.pt"):
                torch.save(t_headsize, name_path + ".headsize.pt")
            if not os.path.exists(name_path + ".trans_matrix.pt"):
                torch.save(t_trans_matrix, name_path + ".trans_matrix.pt")
            if not os.path.exists(name_path + ".original_size.pt"):
                torch.save(t_original_size, name_path + ".original_size.pt")
            if not os.path.exists(name_path + ".bbox.pt"):
                torch.save(t_bbox, name_path + ".bbox.pt")            
            if not os.path.exists(name_path + ".parameters.pt"):
                torch.save(t_parameters, name_path + ".parameters.pt")

        return output

mpii_joint_order = [
    "right ankle",
    "right knee",
    "right hip",
    "left hip",
    "left knee",
    "left ankle",
    "pelvis",
    "thorax",
    "upper neck",
    "head top",
    "right wrist",
    "right elbow",
    "right shoulder",
    "left shoulder",
    "left elbow",
    "left wrist"
]