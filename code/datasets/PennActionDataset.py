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

class PennActionDataset(data.Dataset):

    def __init__(self, root_dir, use_random_parameters=True, transform=None):
        self.root_dir = root_dir
        self.items = sorted(os.listdir(self.root_dir + "frames"))

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
            "baseball_swing": 2,
            "bench_press": 3,
            "bowling": 4,
            "clean_and_jerk": 5,
            "golf_swing": 6,
            "jumping_jacks": 7,
            "jump_rope": 8,
            "pull_ups": 9,
            "push_ups": 10,
            "sit_ups": 11,
            "squats": 12,
            "strumming_guitar": 13,
            "tennis_forehand": 14,
            "tennis_serve": 15
        }

        self.final_size=256

        if use_random_parameters:
            self.angles=np.array(range(-30, 30+1, 5))
            self.scales=np.array([0.7, 1.0, 1.3, 2.5])
            self.channel_power_exponent = 0.01*np.array(range(90, 110+1, 2))
            self.flip_horizontal = np.array([0, 1])
            self.trans_x=np.array(range(-40, 40+1, 5))[0],
            self.trans_y=np.array(range(-40, 40+1, 5))[0],
            self.subsampling=[1, 2]
        else:
            self.angles=np.array([0, 0, 0])
            self.scales=np.array([1., 1., 1.])
            self.flip_horizontal = np.array([0, 0])
            self.channel_power_exponent = None
            self.trans_x=np.array([0., 0., 0.]),
            self.trans_y=np.array([0., 0., 0.]),
            self.subsampling=[1, 1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label_path = self.root_dir + "labels/" + self.items[idx]
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
                joint_frame.append(joint_coordinate)

            poses.append(joint_frame)

        images, normalized_images, poses, trans_matrices = self.preprocess(np.array(images), np.array(poses))

        normalized_pose = poses.copy()
        normalized_pose[:, :,0:2] /= self.final_size

        action = label["action"][0]
        action_1h = np.zeros(16)
        action_1h[self.action_mapping[action]] = 1

        return {
            "action_label": action,
            "action": action_1h,
            "images": images,
            "poses": poses,
            "normalized_frames": normalized_images,
            "normalized_poses": normalized_pose,
            "trans_matrix": trans_matrices
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

        image_width = images.shape[2]
        image_height = images.shape[1]
        window_size = conf_scale * max(image_height, image_width)

        bbox = np.array([
            max(int(image_width / 2) - (window_size / 2), 0), # x1, upper left
            max(int(image_height / 2) - (window_size / 2), 0), # y1, upper left
            min(int(image_width / 2) + (window_size / 2), image_width), # x2, lower right
            min(int(image_height / 2) + (window_size / 2), image_height)  # y2, lower right
        ])

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

            transformed_pose = transform_pose(trans_matrix, pose)

            normalized_image = normalize_channels(image, power_factors=conf_exponents)
            normalized_frames.append(normalized_image)

            final_pose = np.empty((16, 3))
            final_pose[:] = np.nan

            for i in range(13):
                mpii_index = self.mpii_mapping[i][1]
                penn_index = self.mpii_mapping[i][0]

                joint_in_frame =  (0 <= transformed_pose[penn_index][0] <= 256) and (0 <= transformed_pose[penn_index][1] <= 256)
                if joint_in_frame:
                    final_pose[mpii_index, 0:2] = transformed_pose[penn_index]
                else:
                    final_pose[mpii_index, 0:2] = np.array([1e-9, 1e-9])

            final_pose[np.isnan(final_pose)] = 1e-9

            valid_joints = get_valid_joints(final_pose, need_sum=False)[:, 0:2]
            visibility = np.apply_along_axis(np.all, 1, valid_joints)
            final_pose[:, 2] = visibility

            processed_poses.append(final_pose)
            processed_frames.append(image)
            trans_matrices.append(trans_matrix.copy())

        return np.array(processed_frames), np.array(normalized_frames), np.array(processed_poses), np.array(trans_matrices)
