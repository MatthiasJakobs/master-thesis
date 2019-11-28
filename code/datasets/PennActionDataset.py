import torch
import torch.utils.data as data

import random
import math

import os
import re
import glob
import imgaug as ia
import imgaug.augmenters as iaa

import matplotlib.pyplot as plt

import scipy.io as sio
import numpy as np
import pandas as pd

from skimage import io
from skimage.transform import resize

from datasets.BaseDataset import BaseDataset

from deephar.image_processing import center_crop, rotate_and_crop, normalize_channels
from deephar.utils import transform_2d_point, translate, scale, flip_h, superflatten, transform_pose, get_valid_joints, flip_lr_pose, get_bbox_from_pose

actions = [
    "baseball_pitch",
    "baseball_swing",
    "bench_press",
    "bowling",
    "clean_and_jerk",
    "golf_swing",
    "jumping_jacks",
    "jump_rope",
    "pull_ups",
    "push_ups",
    "sit_ups",
    "squats",
    "strumming_guitar",
    "tennis_forehand",
    "tennis_serve"    
]

class PennActionDataset(BaseDataset):

    def __init__(self, root_dir, use_random_parameters=False, transform=None, train=True, val=False, use_saved_tensors=False, augmentation_amount=1, use_gt_bb=False):
        super().__init__(root_dir, use_random_parameters=use_random_parameters, use_saved_tensors=use_saved_tensors, train=train, val=val)        

        self.mpii_mapping = np.array([
            [0, 8],  # head -> upper neck
            [2, 13], # left shoulder
            [1, 12], # right shoulder
            [4, 14], # left elbow
            [3, 11], # right elbow
            [6, 15], # left wrist
            [5, 10], # right wrist
            [8, 3],  # left hip
            [7, 2],  # right hip
            [10, 4],  # left knee
            [9, 1], # right knee
            [12, 5], # left ankle
            [11, 0]  # right ankle
        ])

        self.action_mapping = {
            "baseball_pitch": 0,
            "baseball_swing": 1,
            "bench_press": 2,
            "bowling": 3,
            "clean_and_jerk": 4,
            "golf_swing": 5,
            "jumping_jacks": 6,
            "jump_rope": 7,
            "pull_ups": 8,
            "push_ups": 9,
            "sit_ups": 10,
            "squats": 11,
            "strumming_guitar": 12,
            "tennis_forehand": 13,
            "tennis_serve": 14
        }

        if self.use_random_parameters:
            self.angles=np.array([-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30])
            self.scales=np.array([0.7, 1.0, 1.3])
            self.flip_horizontal = np.array([0, 1])

            self.further_augmentation_prob = 0.5
            self.dropout_size_percent = 0.1
            self.salt_and_pepper = 0.02
            self.motion_blur_angle = list(range(10, 300))

        self.use_gt_bb = use_gt_bb

        self.items = sorted(os.listdir(self.root_dir + "frames"))
        self.indices = []
        self.classes = {}

        self.skip_random = False
        self.augmentation_amount = augmentation_amount

        np.random.seed(None)
        st0 = np.random.get_state()
        np.random.seed(1)

        np.random.shuffle(self.items)

        for i, name in enumerate(self.items):
            label_path = self.root_dir + "labels/" + name
            label = sio.loadmat(label_path)
            raw_label = label["train"][0][0]

            action = self.get_action(label)

            assert raw_label == -1 or raw_label == 1

            train_indicator = bool((raw_label + 1 ) / 2.0)
            if self.train and train_indicator:
                self.indices.append(i)
                if not action in self.classes:
                    self.classes[action] = [ i ]
                else:
                    self.classes[action].append(i)

            if not self.train and not train_indicator:
                self.indices.append(i)
        if self.train:
            for action in actions:
                np.random.shuffle(self.classes[action])

        if self.train:
            self.train_val_split()

        np.random.set_state(st0)

        self.indices = sorted(self.indices)
        #self.items = sorted(self.items)

    # TODO: Identical to JHMDB, need refactor
    def train_val_split(self):
        new_indices = []
        for action in self.classes:
            if self.train and self.val:
                # validation
                lower_limit = 0
                upper_limit = math.ceil(self.val_split_amount * len(self.classes[action]))
            else:
                # train
                lower_limit = math.ceil(self.val_split_amount * len(self.classes[action]))
                upper_limit = len(self.classes[action])
            
            new_indices.extend(self.classes[action][lower_limit:upper_limit])

        self.indices = new_indices

    def get_action(self, label):
        # handling wrong annotations
        action = label["action"][0]
        if action == "bowl":
            action = "bowling"

        if action == "pullup":
            action = "pull_ups"
        
        if action == "pushup":
            action = "push_ups"
        
        if action == "situp":
            action = "sit_ups"        
            
        if action == "squat":
            action = "squats"

        if action == "strum_guitar":
            action = "strumming_guitar"

        return action

    # TODO: Identical to JHMDB, needs refactor
    def map_to_mpii(self, pose):
        final_pose = torch.zeros((16, 3)).float()
        final_pose[:, 0:2] = torch.FloatTensor([-1e9, -1e9])

        for i in range(len(self.mpii_mapping)):
            mpii_index = self.mpii_mapping[i][1].item()
            own_index = self.mpii_mapping[i][0].item()

            joint_in_frame =  (0 <= pose[own_index][0] <= 1) and (0 <= pose[own_index][1] <= 1)
            if joint_in_frame:
                final_pose[mpii_index, 0:2] = pose[own_index]

        return final_pose

    def __getitem__(self, idx):
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
            
            frames = torch.load(self.root_dir + prefix + "images/" + padded_original_image + ".frames.pt")
            actions = torch.load(self.root_dir + prefix + "annotations/" + padded_original_image + ".action_1h.pt")
            poses = torch.load(self.root_dir + prefix + "annotations/" + padded_original_image + ".poses.pt")
            matrices = torch.load(self.root_dir + prefix + "annotations/" + padded_original_image + ".matrices.pt")
            bboxes = torch.load(self.root_dir + prefix + "annotations/" + padded_original_image + ".bbox.pt")
            parameters = torch.load(self.root_dir + prefix + "annotations/" + padded_original_image + ".parameters.pt")

            frames = 2.0 * (frames.float() / 255.0) + 1.0
            poses = poses.float()
            poses[:, :, 0:2] = poses[:,: , 0:2] / 255.0

            t_index = torch.zeros(1).float()
            t_index[0] = idx

            return {
                "action_1h": actions,
                "normalized_frames": frames,
                "normalized_poses": poses,
                "trans_matrices": matrices,
                "parameters": parameters,
                "index": t_index,
                "bbox": bboxes
            }

        label_path = self.root_dir + "labels/" + self.items[self.indices[idx]]

        label = sio.loadmat(label_path)
        images = []
        frame_folder = self.root_dir + "frames/" + self.items[self.indices[idx]] + "/"
        all_images = sorted(os.listdir(frame_folder))

        poses = []
        for i in range(len(all_images)):
            image = io.imread(frame_folder + all_images[i])

            joint_frame = []
            for o in range(13):
                joint_coordinate = [label["x"][i][o], label["y"][i][o], int(label["visibility"][i][o])]
                if joint_coordinate[-1]:
                    joint_frame.append(joint_coordinate)
                else:
                    joint_frame.append([-1e9, -1e9, 0])

            joint_tensor = torch.FloatTensor(joint_frame)
            joint_tensor = joint_tensor.unsqueeze(0)
            _ , valid_sum = get_valid_joints(joint_tensor, need_sum=True)
            if valid_sum.item() > 2:
                images.append(torch.from_numpy(image))
                poses.append(joint_tensor[0, :, 0:2])

        action = self.get_action(label)

        image_height = len(images[0])
        image_width = len(images[0][0])

        if not self.skip_random:
            self.set_augmentation_parameters()
        else:
            self.aug_conf = {}
            self.aug_conf["scale"] = torch.ones(1)
            self.aug_conf["angle"] = torch.zeros(1)
            self.aug_conf["flip"] = torch.zeros(1)
            self.aug_conf["trans_x"] = torch.zeros(1)
            self.aug_conf["trans_y"] = torch.zeros(1)

        processed_frames = []
        processed_poses = []
        trans_matrices = []
        bounding_boxes = []
        original_window_sizes = []

        for frame, pose in zip(images, poses):
            pose_invisible = torch.sum(pose != -1e9) == 0
            if self.use_random_parameters:
                bbox_offset = 40
            else:
                bbox_offset = 30
            bbox_parameters = get_bbox_from_pose(pose, bbox_offset=bbox_offset)
            bbox = bbox_parameters["offset_bbox"]
            window_size = bbox_parameters["offset_window_size"]
            center = bbox_parameters["offset_center"]
            self.original_window_size = bbox_parameters["original_window_size"]
            if self.use_gt_bb and not pose_invisible:
                self.bbox = bbox
                self.window_size = (window_size.float() * self.aug_conf["scale"]).int()
                self.center = center
            else:
                if pose_invisible:
                    print("train {} val {} problem at {}".format(self.train, self.val, idx))
                self.calc_bbox_and_center(image_width, image_height)

            trans_matrix, norm_frame, norm_pose = self.preprocess(frame, pose)

            norm_pose = self.map_to_mpii(norm_pose)
            norm_pose = self.set_visibility(norm_pose)

            if self.aug_conf["flip"]:
                norm_pose = flip_lr_pose(norm_pose)

            bounding_boxes.append(self.bbox.clone().unsqueeze(0))
            processed_poses.append(norm_pose.unsqueeze(0))
            if self.use_random_parameters:
                processed_frames.append(norm_frame.numpy())
            else:
                processed_frames.append(norm_frame.unsqueeze(0))

            trans_matrices.append(trans_matrix.clone().unsqueeze(0))
            original_window_sizes.append(self.original_window_size.clone().unsqueeze(0))

        # further augmentation, beyond Luvizon et al.
        if self.use_random_parameters:
            processed_frames = list(map(lambda x: (x + 1) / 2.0, processed_frames))
            seq = iaa.Sequential(
                [
                    #iaa.Sometimes(1.0, iaa.MotionBlur(k=3, angle=self.motion_blur_angle)),
                    iaa.Sequential([
                        iaa.Sometimes(self.further_augmentation_prob, iaa.CoarseDropout((0.05 * self.aug_conf["scale"]).item(), size_percent=(self.dropout_size_percent * self.aug_conf["scale"]).item())),
                        iaa.Sometimes(self.further_augmentation_prob, iaa.SaltAndPepper((self.salt_and_pepper * self.aug_conf["scale"]).item())),
                    ], random_order=True)
                ]
            , random_order=False)
            np_processed_frames = seq(images=processed_frames)
            processed_frames = list(map(lambda x: 2.0 * torch.FloatTensor(np.expand_dims(x, axis=0)) - 1, np_processed_frames))


        original_window_sizes = torch.cat(original_window_sizes)
        frames = torch.cat(processed_frames)
        poses = torch.cat(processed_poses)
        trans_matrices = torch.cat(trans_matrices)
        t_bounding_boxes = torch.cat(bounding_boxes)

        frames = frames.permute(0, 3, 1, 2)

        action_1h = torch.zeros(15).float()
        action_1h[self.action_mapping[action]] = 1

        t_sequence_length = torch.ByteTensor([len(frames)])

        t_index = torch.zeros(1).float()
        t_index[0] = idx

        t_parameters = torch.zeros(5).float()
        t_parameters[0] = self.aug_conf["scale"]
        t_parameters[1] = float(self.aug_conf["angle"])
        t_parameters[2] = float(self.aug_conf["flip"])
        t_parameters[3] = float(self.aug_conf["trans_x"])
        t_parameters[4] = float(self.aug_conf["trans_y"])

        return {
            "action_1h": action_1h,
            "action_label": action,
            "normalized_frames": frames,
            "normalized_poses": poses,
            "sequence_length": t_sequence_length,
            "trans_matrices": trans_matrices,
            "parameters": t_parameters,
            "index": t_index,
            "original_window_size": original_window_sizes,
            "bbox": t_bounding_boxes,
            "frame_folder": frame_folder
        }
