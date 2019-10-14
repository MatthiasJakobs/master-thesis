import torch
import torch.utils.data as data

import random
import math

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

    def __init__(self, root_dir, use_random_parameters=False, transform=None, train=True, val=False):
        super().__init__(root_dir, use_random_parameters=use_random_parameters, train=train, val=val)        

        self.mpii_mapping = np.array([
            [0, 8],  # head -> upper neck
            [1, 13], # left shoulder
            [2, 12], # right shoulder
            [3, 14], # left elbow
            [4, 11], # right elbow
            [5, 15], # left wrist
            [6, 10], # right wrist
            [7, 3],  # left hip
            [8, 2],  # right hip
            [9, 4],  # left knee
            [10, 1], # right knee
            [11, 5], # left ankle
            [12, 0]  # right ankle
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
            self.angles=np.array(range(-30, 30+1, 5))
            self.scales=np.array([0.7, 1.0, 1.3, 2.5])
            self.flip_horizontal = np.array([0, 1])

        self.items = sorted(os.listdir(self.root_dir + "frames"))
        self.indices = []
        self.classes = {}

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
            self.train_val_split()

        np.random.set_state(st0)

        self.indices = sorted(self.indices)
        self.items = sorted(self.items)

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


    def calc_gt_bb(self, pose, offset=None):
        min_x = 1e10
        max_x = -1e10
        min_y = 1e10
        max_y = -1e10

        bbox = torch.IntTensor(4)

        for joint in poses:
            x = joint[0]
            y = joint[1]

            if x < min_x:
                min_x = x

            if x > max_x:
                max_x = x

            if y < min_y:
                min_y = y

            if y > max_y:
                max_y = y

        bbox[0] = min_x
        bbox[1] = min_y
        bbox[2] = max_x
        bbox[3] = min_y

        bbox_width = torch.abs(bbox[0] - bbox[2]).item()
        bbox_height = torch.abs(bbox[1] - bbox[3]).item()

        if offset is not None:
            window_size = torch.IntTensor([max(bbox_height, bbox_width) + offset, max(bbox_height, bbox_width) + offset])
        else:
            window_size = torch.IntTensor([max(bbox_height, bbox_width), max(bbox_height, bbox_width)])
        self.window_size = window_size
        center = torch.IntTensor([
            bbox[2] - bbox_width / 2,
            bbox[3] - bbox_height / 2
        ])
        self.bbox = bbox
        self.center = center

    def __getitem__(self, idx):
        label_path = self.root_dir + "labels/" + self.items[self.indices[idx]]

        label = sio.loadmat(label_path)
        images = []
        frame_folder = self.root_dir + "frames/" + self.items[self.indices[idx]] + "/"
        all_images = sorted(os.listdir(frame_folder))

        poses = []
        for i in range(len(all_images)):
            image = io.imread(frame_folder + all_images[i])
            images.append(torch.from_numpy(image))

            joint_frame = []
            for o in range(13):
                joint_coordinate = [label["x"][i][o], label["y"][i][o]]
                visibility = bool(label["visibility"][i][o])
                if visibility:
                    joint_frame.append(joint_coordinate)
                else:
                    joint_frame.append([-1e9, -1e9])

            poses.append(torch.FloatTensor(joint_frame))

        action = self.get_action(label)

        image_height = len(images[0])
        image_width = len(images[0][0])

        self.set_augmentation_parameters()

        processed_frames = []
        processed_poses = []
        trans_matrices = []
        bounding_boxes = []


        for frame, pose in zip(images, poses):
            if self.train:
                self.calc_gt_bb(pose)
            else:
                self.calc_bbox_and_center(image_width, image_height)

            trans_matrix, norm_frame, norm_pose = self.preprocess(frame, pose)

            norm_pose = self.map_to_mpii(norm_pose)
            norm_pose = self.set_visibility(norm_pose)

            if self.aug_conf["flip"]:
                norm_pose = flip_lr_pose(norm_pose)

            bounding_boxes.append(self.bbox.clone().unsqueeze(0))
            processed_poses.append(norm_pose.unsqueeze(0))
            processed_frames.append(norm_frame.unsqueeze(0))
            trans_matrices.append(trans_matrix.clone().unsqueeze(0))


        frames = torch.cat(processed_frames)
        poses = torch.cat(processed_poses)
        trans_matrices = torch.cat(trans_matrices)
        t_bounding_boxes = torch.cat(bounding_boxes)

        frames = frames.permute(0, 3, 1, 2)

        action_1h = torch.zeros(15).float()
        action_1h[self.action_mapping[action]] = 1

        # t_sequence_length = torch.ByteTensor([number_of_frames])

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
            # "sequence_length": t_sequence_length,
            "trans_matrices": trans_matrices,
            "parameters": t_parameters,
            "index": t_index,
            "bbox": t_bounding_boxes
        }
