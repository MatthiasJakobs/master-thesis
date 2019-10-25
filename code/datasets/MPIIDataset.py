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

from datasets.BaseDataset import BaseDataset

from deephar.image_processing import center_crop, rotate_and_crop, normalize_channels
from deephar.utils import transform_2d_point, translate, scale, flip_h, superflatten, transform_pose, get_valid_joints, flip_lr_pose

class MPIIDataset(BaseDataset):

    '''
    What is (probably) happening with the train / val split of luvizon:
        - They load training data only and extract each person. Afterwards, they split it up
          into train / val and save it.
        - When I do it like below, almost exactly size(me) = size(train_luvizon) + size(val_luvizon)
    '''

    def __init__(self, root_dir, transform=None, use_random_parameters=True, train=True, val=False, use_saved_tensors=False, augmentation_amount=1):
        super().__init__(root_dir, use_random_parameters=use_random_parameters, use_saved_tensors=use_saved_tensors, train=train, val=val)

        assert train == True # no test set available

        assert "annotations.mat" in os.listdir(self.root_dir)

        annotations = sio.loadmat(self.root_dir + "annotations")["RELEASE"]

        train_binary = annotations["img_train"][0][0][0]
        train_indeces = np.where(np.array(train_binary))[0]

        assert augmentation_amount > 0
        self.augmentation_amount = augmentation_amount

        if self.use_random_parameters:
            self.prefix = "rand_"
        else:
            self.prefix = ""

        if not os.path.exists(root_dir + self.prefix + "tensors"):
            os.makedirs(root_dir + self.prefix + "tensors")

        if use_random_parameters:
            self.angles=np.array(range(-40, 40+1, 5))
            self.scales=np.array([0.7, 1., 1.3])
            self.flip_horizontal = np.array([0, 1])

        self.labels = []
        missing_annnotation_count = 0

        self.skip_random = False

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
        
        self.items = self.labels

        self.train_val_split()

    def train_val_split(self):
        np.random.seed(None)
        st0 = np.random.get_state()
        np.random.seed(1)

        #np.random.shuffle(self.items)

        val_limit = int(self.val_split_amount * len(self.items))

        all_indices = list(range(len(self.items)))
        np.random.shuffle(all_indices)

        if self.train and self.val:
            self.indices = all_indices[0:val_limit]
        else:
            self.indices = all_indices[val_limit:]

        np.random.set_state(st0)

        self.indices = sorted(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label = self.items[self.indices[idx]]
        output = {}

        if self.use_saved_tensors:

            if self.val:
                train_test_folder = "val/"
            else:
                train_test_folder = "train/"

            if self.use_random_parameters:
                dice_roll = random.randint(0, self.augmentation_amount)
                if dice_roll == 0:
                    prefix = ""
                else:
                    prefix = "rand{}_".format(dice_roll)
            else:
                prefix = ""

            original_image = self.indices[idx]
            padded_original_image = str(original_image).zfill(8)

            name_path = self.root_dir + train_test_folder + prefix
            
            t_filepath = torch.load(name_path + "annotations/" + padded_original_image + ".image_path.pt")
            t_normalized_image = torch.load(name_path + "images/" + padded_original_image + ".frame.pt")
            t_normalized_pose = torch.load(name_path + "annotations/" + padded_original_image + ".pose.pt")
            t_headsize = torch.load(name_path + "annotations/" + padded_original_image + ".headsize.pt")
            t_trans_matrix = torch.load(name_path + "annotations/" + padded_original_image + ".matrix.pt")
            t_bbox = torch.load(name_path + "annotations/" + padded_original_image + ".bbox.pt")
            t_parameters = torch.load(name_path + "annotations/" + padded_original_image + ".parameters.pt")

            t_normalized_image = 2.0 * (t_normalized_image.float() / 255.0) + 1.0
            t_normalized_pose = t_normalized_pose.float()
            t_normalized_pose[:, 0:2] = t_normalized_pose[:, 0:2] / 255.0

            output["image_path"] = t_filepath
            output["normalized_image"] = t_normalized_image
            output["normalized_pose"] = t_normalized_pose
            #output["original_pose"] = t_original_pose
            output["head_size"] = t_headsize
            output["trans_matrix"] = t_trans_matrix
            #output["original_size"] = t_original_size
            output["bbox"] = t_bbox
            output["parameters"] = t_parameters         

            return output

        full_image_path = self.root_dir + "images/" + label["image_name"]
        image = io.imread(full_image_path)

        if not self.skip_random:
            self.set_augmentation_parameters()
        else:
            self.aug_conf = {}
            self.aug_conf["scale"] = torch.ones(1)
            self.aug_conf["angle"] = torch.zeros(1)
            self.aug_conf["flip"] = torch.zeros(1)
            self.aug_conf["trans_x"] = torch.zeros(1)
            self.aug_conf["trans_y"] = torch.zeros(1)

        new_scale = label["scale"] * 1.25 # magic value
        new_objpose = np.array([label["obj_pose"][0], label["obj_pose"][1] + 12 * new_scale]) # magic values, no idea where they are comming from

        window_size = new_scale * self.aug_conf["scale"].item() * 200

        self.bbox = torch.IntTensor([
            new_objpose[0] - window_size / 2, # x1, upper left
            new_objpose[1] - window_size / 2, # y1, upper left
            new_objpose[0] + window_size / 2, # x2, lower right
            new_objpose[1] + window_size / 2  # y2, lower right
        ])

        self.center = torch.from_numpy(new_objpose).float()
        self.window_size = [window_size, window_size]

        pose = torch.IntTensor([label["pose"]["x"], label["pose"]["y"]]).t()
        frame = torch.from_numpy(image)

        # preprocess
        trans_matrix, norm_frame, norm_pose = self.preprocess(frame, pose)

        mapped_pose = torch.FloatTensor(16, 3)
        mapped_pose[:, 0:2] = -1e9

        for it, joint_index in enumerate(label["pose"]["ids"]):
            x = norm_pose[it, 0]
            y = norm_pose[it, 1]

            if x < 0 or y < 0 or x > 1 or y > 1:
                mapped_pose[joint_index, 0] = -1e9
                mapped_pose[joint_index, 1] = -1e9
            else:
                mapped_pose[joint_index, 0] = x
                mapped_pose[joint_index, 1] = y

            mapped_pose[joint_index, 2] = float(label["pose"]["visible"][it])

        norm_pose = mapped_pose
        norm_pose = self.set_visibility(norm_pose)

        if self.aug_conf["flip"]:
            norm_pose = flip_lr_pose(norm_pose)


        # calculating head size for pckh (according to paper)
        head_point_upper = torch.FloatTensor([label["head"][0], label["head"][1]])
        head_point_lower = torch.FloatTensor([label["head"][2], label["head"][3]])
        head_size = 0.6 * np.linalg.norm(head_point_upper - head_point_lower)

        headsize = torch.from_numpy(np.array([head_size])).float()

        norm_frame = norm_frame.permute(2, 0, 1)
        
        t_parameters = torch.zeros(5).float()
        t_parameters[0] = self.aug_conf["scale"]
        t_parameters[1] = float(self.aug_conf["angle"])
        t_parameters[2] = float(self.aug_conf["flip"])
        t_parameters[3] = float(self.aug_conf["trans_x"])
        t_parameters[4] = float(self.aug_conf["trans_y"])

        image_number = int(full_image_path[-13:-4])
        
        image_number_np = np.array([image_number])
        t_filepath = torch.from_numpy(image_number_np).int()

        output["bbox"] = self.bbox
        output["image_path"] = t_filepath
        output["normalized_image"] = norm_frame
        output["normalized_pose"] = norm_pose
        output["head_size"] = headsize
        output["trans_matrix"] = trans_matrix
        output["parameters"] = t_parameters

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