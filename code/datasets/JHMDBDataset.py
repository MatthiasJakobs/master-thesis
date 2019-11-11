import torch
import torch.utils.data as data

import random
import math
import os
import re
import glob
import csv

import scipy.io as sio
import numpy as np
import pandas as pd

from skimage import io
from skimage.transform import resize

from datasets.BaseDataset import BaseDataset

from deephar.image_processing import center_crop, rotate_and_crop, normalize_channels
from deephar.utils import transform_2d_point, translate, scale, flip_h, superflatten, transform_pose, get_valid_joints, flip_lr_pose

actions = [
    "brush_hair",
    "catch",
    "clap",
    "climb_stairs",
    "golf",
    "jump",
    "kick_ball",
    "pick",
    "pour",
    "pullup",
    "push",
    "run",
    "shoot_ball",
    "shoot_bow",
    "shoot_gun",
    "sit",
    "stand",
    "swing_baseball",
    "throw",
    "walk",
    "wave"
]

class JHMDBDataset(BaseDataset):

    def __init__(self, root_dir, transform=None, use_random_parameters=False, use_saved_tensors=False, split=1, train=True, val=True, use_gt_bb=False):

        super().__init__(root_dir, use_random_parameters=use_random_parameters, use_saved_tensors=use_saved_tensors, train=train, val=val)

        self.split = split
        self.clip_length = 40
        self.use_gt_bb = use_gt_bb

        split_file_paths = "{}splits/*_test_split{}.txt".format(self.root_dir, split)
        split_files = glob.glob(split_file_paths)

        self.items = glob.glob(self.root_dir + "*/*")
        self.items = [item for item in self.items if not "/splits/" in item]
        self.items = [item for item in self.items if not "/puppet_mask/" in item]

        key = 1 if train else 2

        self.indices = []
        self.classes = {}

        np.random.seed(None)
        st0 = np.random.get_state()
        np.random.seed(1)

        np.random.shuffle(self.items)

        for train_test_file in split_files:
            with open(train_test_file) as csv_file:
                reader = csv.reader(csv_file, delimiter=" ")
                for row in reader:
                    if int(row[1]) == key:
                        clip_name = row[0][:-4]
                        for idx, name in enumerate(self.items):
                            path_split = name.split("/")
                            action = path_split[-2]
                            if path_split[-1] == clip_name:
                                self.indices.append(idx)

                                if not action in self.classes:
                                    self.classes[action] = [ idx ]
                                else:
                                    self.classes[action].append(idx)

        if train:
            self.train_val_split()

        np.random.set_state(st0)

        self.indices = sorted(self.indices)
        self.items = sorted(self.items)

        if self.use_random_parameters:
            self.angles=torch.IntTensor([-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30])
            self.scales=torch.FloatTensor([0.7, 1.3])
            self.flip_horizontal = torch.ByteTensor([0, 1])

        # REMEMBER: 15 Joint positions, MPII has 16!
        self.mpii_mapping = torch.ByteTensor([
            [0, 8],  # neck -> upper neck
            [1, 6], # belly -> pelvis
            [2, 9], # face -> head_top
            [3, 12], # left shoulder
            [4, 13], # right shoulder
            [5, 2], # left hip
            [6, 3], # right hip
            [7, 11],  # left elbow
            [8, 14],  # right elbow
            [9, 1],  # left knee
            [10, 4], # right knee
            [11, 10], # left wrist
            [12, 15],  # right wrist
            [13, 0], # left ankle
            [14, 5]  # right ankle
        ])

        self.joint_mapping = [
            "neck",
            "belly",
            "face",
            "right shoulder",
            "left  shoulder",
            "right hip",
            "left  hip",
            "right elbow",
            "left elbow",
            "right knee",
            "left knee",
            "right wrist",
            "left wrist",
            "right ankle",
            "left ankle"
        ]

        self.action_mapping = {
            "brush_hair": 0,
            "catch": 1,
            "clap": 2,
            "climb_stairs": 3,
            "golf": 4,
            "jump": 5,
            "kick_ball": 6,
            "pick": 7,
            "pour": 8,
            "pullup": 9,
            "push": 10,
            "run": 11,
            "shoot_ball": 12,
            "shoot_bow": 13,
            "shoot_gun": 14,
            "sit": 15,
            "stand": 16,
            "swing_baseball": 17,
            "throw": 18,
            "walk": 19,
            "wave": 20
        }

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

    def apply_padding(self, trans_matrices, frames, poses, bboxes, original_window_sizes):
        number_of_frames = len(frames)
        if number_of_frames < self.clip_length:

            # padding for images
            desired_shape = frames[0].shape
            blanks = torch.zeros((self.clip_length - number_of_frames, desired_shape[0], desired_shape[1], desired_shape[2])).float()
            frames = torch.cat((frames, blanks))

            # padding for poses
            assert poses.shape[1:] == (16, 3)
            blanks = torch.zeros((self.clip_length - number_of_frames, 16, 3)).float()
            poses = torch.cat((poses, blanks))

            # padding for matrices
            assert trans_matrices.shape[1:] == (3, 3)
            blanks = torch.zeros((self.clip_length - number_of_frames, 3, 3)).float()
            trans_matrices = torch.cat((trans_matrices, blanks))

            # padding for bboxes
            assert bboxes.shape[1] == 4, print(bboxes.shape)
            blanks = torch.zeros((self.clip_length - number_of_frames, 4)).int()
            bboxes = torch.cat((bboxes, blanks))

            # padding for original_window_sizes
            assert original_window_sizes.shape[1] == 2, print(original_window_sizes.shape)
            blanks = torch.zeros((self.clip_length - number_of_frames, 2)).int()
            original_window_sizes = torch.cat((original_window_sizes, blanks))

        return trans_matrices, frames, poses, bboxes, original_window_sizes

    def create_puppet_mask(self, idx):
        item_path = self.items[self.indices[idx]]
        relative_path_split = item_path[len(self.root_dir):].split("/")
        action = relative_path_split[0]

        puppet_mask_file = self.root_dir + "puppet_mask/" + relative_path_split[0] + "/" + relative_path_split[1] + "/puppet_mask"
        puppet_mask = sio.loadmat(puppet_mask_file)
        binary_masks = torch.from_numpy(puppet_mask["part_mask"]).int()
        binary_masks = binary_masks.permute(2, 1, 0)
        puppet_corners = torch.IntTensor(len(binary_masks), 4)

        puppet_width = binary_masks.size()[1]
        puppet_height = binary_masks.size()[2]

        for i in range(len(binary_masks)):
            min_y = puppet_height
            max_y = 0
            min_x = puppet_width
            max_x = 0
            for y in range(puppet_height):
                for x in range(puppet_width):
                    if binary_masks[i, x, y] == 1:
                        if x < min_x:
                            min_x = x
                        if x > max_x:
                            max_x = x
                        if y < min_y:
                            min_y = y
                        if y > max_y:
                            max_y = y

            puppet_corners[i][0] = min_x
            puppet_corners[i][1] = max_x
            puppet_corners[i][2] = min_y
            puppet_corners[i][3] = max_y

        out_file = self.root_dir + "puppet_mask/" + relative_path_split[0] + "/" + relative_path_split[1] + "/puppet_tensor.pt"
        torch.save(puppet_corners, out_file)

    def __getitem__(self, idx):
        item_path = self.items[self.indices[idx]]
        relative_path_split = item_path[len(self.root_dir):].split("/")
        action = relative_path_split[0]

        puppet_mask_file = self.root_dir + "puppet_mask/" + relative_path_split[0] + "/" + relative_path_split[1] + "/puppet_tensor.pt"
        puppet_corners = torch.load(puppet_mask_file)

        label = sio.loadmat(item_path + "/joint_positions")
        poses = torch.from_numpy(label["pos_img"].T).float()

        all_frames = sorted(glob.glob(item_path + "/*.png"))
        images = []
        for image_path in all_frames:
            images.append(torch.from_numpy(io.imread(image_path)).int())

        image_height = len(images[0])
        image_width = len(images[0][0])

        self.set_augmentation_parameters()

        #self.calc_bbox_and_center(image_width, image_height)

        processed_frames = []
        processed_poses = []
        trans_matrices = []
        bounding_boxes = []
        original_window_sizes = []

        current_frame = 0
        for frame, pose in zip(images, poses):
            bbox = torch.IntTensor(4)
            bbox[0] = puppet_corners[current_frame, 0]
            bbox[1] = puppet_corners[current_frame, 2]
            bbox[2] = puppet_corners[current_frame, 1]
            bbox[3] = puppet_corners[current_frame, 3]
            
            bbox_width = torch.abs(bbox[0] - bbox[2]).item()
            bbox_height = torch.abs(bbox[1] - bbox[3]).item()
            self.original_window_size = torch.IntTensor([max(bbox_height, bbox_width), max(bbox_height, bbox_width)])

            if self.use_random_parameters:
                offset = 40
            else:
                offset = 30

            if self.use_gt_bb:
                self.calc_bbox_and_center(image_width, image_height, pre_bb=bbox, offset=offset)
            else:
                self.calc_bbox_and_center(image_width, image_height)

            current_frame = current_frame + 1

            trans_matrix, norm_frame, norm_pose = self.preprocess(frame, pose)

            norm_pose = self.map_to_mpii(norm_pose)
            norm_pose = self.set_visibility(norm_pose)

            if self.aug_conf["flip"]:
                norm_pose = flip_lr_pose(norm_pose)

            bounding_boxes.append(self.bbox.clone().unsqueeze(0))
            processed_poses.append(norm_pose.unsqueeze(0))
            processed_frames.append(norm_frame.unsqueeze(0))
            trans_matrices.append(trans_matrix.clone().unsqueeze(0))
            original_window_sizes.append(self.original_window_size.clone().unsqueeze(0))

        number_of_frames = len(processed_frames)

        frames = torch.cat(processed_frames)
        poses = torch.cat(processed_poses)
        trans_matrices = torch.cat(trans_matrices)
        t_bounding_boxes = torch.cat(bounding_boxes)
        original_window_sizes = torch.cat(original_window_sizes)

        trans_matrices, frames, poses, t_bounding_boxes, original_window_sizes = self.apply_padding(trans_matrices, frames, poses, t_bounding_boxes, original_window_sizes)

        frames = frames.permute(0, 3, 1, 2)

        action_1h = torch.zeros(21).float()
        action_1h[self.action_mapping[action]] = 1

        t_sequence_length = torch.ByteTensor([number_of_frames])

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
            "original_window_size": original_window_sizes,
            "index": t_index,
            "bbox": t_bounding_boxes
        }
