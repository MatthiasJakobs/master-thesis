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

class PennActionDataset(data.Dataset):

    def __init__(self, root_dir, use_random_parameters=False, transform=None, train=True):
        self.root_dir = root_dir
        self.all_items = sorted(os.listdir(self.root_dir + "frames"))

        self.train = train

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

        self.final_size=256

        if use_random_parameters:
            self.angles=np.array(range(-30, 30+1, 5))
            self.scales=np.array([0.7, 1.0, 1.3, 2.5])
            self.channel_power_exponent = 0.01*np.array(range(90, 110+1, 2))
            self.flip_horizontal = np.array([0, 1])
            self.trans_x=np.arange(start=-40, stop=40+1, step=5)
            self.trans_y=np.arange(start=-40, stop=40+1, step=5)
            self.subsampling=[1, 2]
        else:
            self.angles=np.array([0, 0, 0])
            self.scales=np.array([1., 1., 1.])
            self.flip_horizontal = np.array([0, 0])
            self.channel_power_exponent = None
            self.trans_x=np.array([0., 0., 0.])
            self.trans_y=np.array([0., 0., 0.])
            self.subsampling=[1, 1]

        self.items = []
        self.indices = []

        for i in range(len(self.all_items)):
            label_path = self.root_dir + "labels/" + str(i+1).zfill(4)
            label = sio.loadmat(label_path)
            raw_label = label["train"][0][0]

            assert raw_label == -1 or raw_label == 1

            train_indicator = bool((raw_label + 1 ) / 2.0)
            if self.train == train_indicator:
                self.items.append(self.all_items[i])
                self.indices.append(i+1)
            

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label_path = self.root_dir + "labels/" + self.items[idx]
        if self.items[idx] == 435:
            print("here is something wrong!")

        label = sio.loadmat(label_path)
        images = []
        frame_folder = self.root_dir + "frames/" + self.items[idx] + "/"
        all_images = sorted(os.listdir(frame_folder))

        poses = []
        for i in range(len(all_images)):
            image = io.imread(frame_folder + all_images[i])
            images.append(image)

            joint_frame = []
            for o in range(13):
                joint_coordinate = [label["x"][i][o], label["y"][i][o]]
                visibility = bool(label["visibility"][i][o])
                if visibility:
                    joint_frame.append(joint_coordinate)
                else:
                    joint_frame.append([-1e9, -1e9])

            poses.append(joint_frame)

        images, normalized_images, poses, trans_matrices = self.preprocess(np.array(images), np.array(poses))

        normalized_pose = poses.copy()
        normalized_pose[:, :,0:2]

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

        t_action_1h = torch.zeros(15)
        t_action_1h[self.action_mapping[action]] = 1

        t_index = torch.zeros(1)
        t_index[0] = idx

        t_parameters = torch.zeros(5).float()
        t_parameters[0] = self.test["scale"]
        t_parameters[1] = float(self.test["angle"])
        t_parameters[2] = float(self.test["flip"])
        t_parameters[3] = float(self.test["trans_x"])
        t_parameters[4] = float(self.test["trans_y"])

        t_bbox = torch.from_numpy(self.bbox).float()

        t_normalized_frames = torch.from_numpy(normalized_images).float()
        t_normalized_poses = torch.from_numpy(normalized_pose).float()
        t_trans_matrices = torch.from_numpy(trans_matrices).float()

        return {
            "action_1h": t_action_1h,
            "normalized_frames": t_normalized_frames,
            "normalized_poses": t_normalized_poses,
            "trans_matrices": t_trans_matrices,
            "bbox": t_bbox,
            "parameters": t_parameters
        }

    def preprocess(self, images, poses):
        conf_scale = self.scales[np.random.randint(0, len(self.scales))]
        conf_angle = self.angles[np.random.randint(0, len(self.angles))]
        conf_flip = self.flip_horizontal[np.random.randint(0, len(self.flip_horizontal))]
        conf_subsample = self.subsampling[np.random.randint(0, len(self.subsampling))]
        conf_trans_x = self.trans_x[np.random.randint(0, len(self.trans_x))]
        conf_trans_y = self.trans_y[np.random.randint(0, len(self.trans_y))]

        if self.channel_power_exponent is not None:
            conf_exponents = np.array([
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))],
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))],
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))]
            ])
        else:
            conf_exponents = None

        self.test = {}
        self.test["angle"] = conf_angle
        self.test["scale"] = conf_scale
        self.test["flip"] = conf_flip
        self.test["trans_x"] = conf_trans_x
        self.test["trans_y"] = conf_trans_y

        image_width = images.shape[2]
        image_height = images.shape[1]
        window_size = conf_scale * max(image_height, image_width)

        bbox = np.array([
            int(image_width / 2) - (window_size / 2), # x1, upper left
            int(image_height / 2) - (window_size / 2), # y1, upper left
            int(image_width / 2) + (window_size / 2), # x2, lower right
            int(image_height / 2) + (window_size / 2)  # y2, lower right
        ])

        self.bbox = bbox

        bbox_width = int(abs(bbox[0] - bbox[2]))
        bbox_height = int(abs(bbox[1] - bbox[3]))
        window_size = np.array([bbox_width, bbox_height])
        center = np.array([
            int(bbox[2] - int(bbox_width / 2)),
            int(bbox[3] - int(bbox_height / 2))
        ])

        assert bbox_width >= 32 and bbox_height >= 32

        center += np.array(conf_scale * np.array([conf_trans_x, conf_trans_y])).astype(int)

        processed_frames = []
        processed_poses = []
        normalized_frames = []
        visibility = []
        trans_matrices = []

        #TODO: transx, transy
        for frame, pose in zip(images, poses):
            trans_matrix, image = rotate_and_crop(frame, conf_angle, center, window_size)
            size_after_rotate = np.array([image.shape[1], image.shape[0]])

            image = resize(image, (256, 256), preserve_range=True)
            trans_matrix = scale(trans_matrix, 256 / size_after_rotate[0], 256 / size_after_rotate[1])

            #TODO: subsampling

            # randomly flip horizontal
            if conf_flip:
                image = np.fliplr(image)

                trans_matrix = translate(trans_matrix, -image.shape[1] / 2, -image.shape[0] / 2)
                trans_matrix = flip_h(trans_matrix)
                trans_matrix = translate(trans_matrix, image.shape[1] / 2, image.shape[0] / 2)

            trans_matrix = scale(trans_matrix, 1.0 / self.final_size, 1.0 / self.final_size)

            transformed_pose = transform_pose(trans_matrix, pose)

            normalized_image = normalize_channels(image, power_factors=conf_exponents)
            normalized_frames.append(normalized_image)

            final_pose = np.empty((16, 3))
            final_pose[:] = np.nan

            for i in range(13):
                mpii_index = self.mpii_mapping[i][1]
                penn_index = self.mpii_mapping[i][0]

                joint_in_frame =  (0 <= transformed_pose[penn_index][0] <= 1) and (0 <= transformed_pose[penn_index][1] <= 1)
                if joint_in_frame:
                    final_pose[mpii_index, 0:2] = transformed_pose[penn_index]
                else:
                    final_pose[mpii_index, 0:2] = np.array([-1e9, -1e9])

            final_pose[np.isnan(final_pose)] = -1e9

            valid_joints = get_valid_joints(final_pose, need_sum=False)[:, 0:2]
            visibility = np.apply_along_axis(np.all, 1, valid_joints)
            final_pose[:, 2] = visibility

            processed_poses.append(final_pose)
            processed_frames.append(image)
            trans_matrices.append(trans_matrix.copy())

        return np.array(processed_frames), np.array(normalized_frames), np.array(processed_poses), np.array(trans_matrices)
