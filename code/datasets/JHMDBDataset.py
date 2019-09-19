import torch
import torch.utils.data as data

import random

import os
import re
import glob
import csv

import scipy.io as sio
import numpy as np
import pandas as pd

from skimage import io
from skimage.transform import resize

from deephar.image_processing import center_crop, rotate_and_crop, normalize_channels
from deephar.utils import transform_2d_point, translate, scale, flip_h, superflatten, transform_pose, get_valid_joints

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

class JHMDBDataset(data.Dataset):

    def __init__(self, root_dir, transform=None, use_random_parameters=False, use_saved_tensors=False, split=1, train=True):
        self.root_dir = root_dir
        self.train = train
        self.split = split
        
        split_file_paths = self.root_dir + "splits/*_test_split" + str(split) + ".txt"
        split_files = glob.glob(split_file_paths)

        self.all_images = sorted(glob.glob(self.root_dir + "*/*"))
        
        clips = []
        all_frames = []
        key = 1 if train else 2

        self.indices = []

        for train_test_file in split_files:
            with open(train_test_file) as csv_file:
                reader = csv.reader(csv_file, delimiter=" ")
                for row in reader:
                    if int(row[1]) == key:
                        clip_name = row[0][:-4]
                        for idx, name in enumerate(self.all_images):
                            if name.split("/")[-1] == clip_name:
                                self.indices.append(idx)
                        clips.append(clip_name)

        self.indices = sorted(self.indices)

        for clip_name in clips:
            frames = glob.glob(self.root_dir + "*/" + clip_name)
            all_frames.extend(frames)

        self.items = sorted(all_frames)
        assert len(self.items) == len(self.indices)

        self.final_size = 255

        if use_random_parameters:
            self.angles=np.array(range(-30, 30+1, 5))
            self.scales=np.array([0.7, 1.0, 1.3])
            self.channel_power_exponent = 0.01*np.array(range(90, 110+1, 2))
            self.flip_horizontal = np.array([0, 1])
            self.trans_x=np.array(range(-40, 40+1, 5))
            self.trans_y=np.array(range(-40, 40+1, 5))
            self.subsampling=[1, 2]
        else:
            self.angles=np.array([0, 0, 0])
            self.scales=np.array([1., 1., 1.])
            self.flip_horizontal = np.array([0, 0])
            self.channel_power_exponent = None
            self.trans_x=np.array([0., 0., 0.])
            self.trans_y=np.array([0., 0., 0.])
            self.subsampling=[1, 1]

        self.mpii_mapping = np.array([
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

    def __len__(self):
        return len(self.items)

    def preprocess(self, images, poses):
        conf_scale = self.scales[np.random.randint(0, len(self.scales))]
        conf_angle = self.angles[np.random.randint(0, len(self.angles))]
        conf_flip = self.flip_horizontal[np.random.randint(0, len(self.flip_horizontal))]
        #conf_subsample = self.subsampling[np.random.randint(0, len(self.subsampling))]
        conf_trans_x = self.trans_x[np.random.randint(0, len(self.trans_x))]
        conf_trans_y = self.trans_y[np.random.randint(0, len(self.trans_y))]

        self.test = {}
        self.test["scale"] = conf_scale
        self.test["angle"] = conf_angle
        self.test["flip"] = conf_flip
        self.test["trans_x"] = conf_trans_x
        self.test["trans_y"] = conf_trans_y

        '''if self.channel_power_exponent is not None:
            conf_exponents = np.array([
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))],
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))],
                self.channel_power_exponent[np.random.randint(0, len(self.channel_power_exponent))]
            ])
        else:
            conf_exponents = None
        '''
        conf_exponents = None
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
        trans_matrices = []
        normalized_frames = []
        visibility = []

        #TODO: transx, transy
        for frame, pose in zip(images, poses):
            trans_matrix, image = rotate_and_crop(frame, conf_angle, center, window_size)
            size_after_rotate = np.array([image.shape[1], image.shape[0]])

            image = resize(image, (self.final_size, self.final_size), preserve_range=True)
            trans_matrix = scale(trans_matrix, self.final_size / size_after_rotate[0], self.final_size / size_after_rotate[1])

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

            for i in range(len(self.mpii_mapping)):
                mpii_index = self.mpii_mapping[i][1]
                jhmdb_index = self.mpii_mapping[i][0]

                joint_in_frame =  (0 <= transformed_pose[jhmdb_index][0] <= 1) and (0 <= transformed_pose[jhmdb_index][1] <= 1)
                if joint_in_frame:
                    final_pose[mpii_index, 0:2] = transformed_pose[jhmdb_index]
                else:
                    final_pose[mpii_index, 0:2] = np.array([-1e9, -1e9])

            final_pose[np.isnan(final_pose)] = -1e9

            valid_joints = get_valid_joints(final_pose, need_sum=False)[:, 0:2]
            visibility = np.apply_along_axis(np.all, 1, valid_joints)
            final_pose[:, 2] = visibility

            processed_poses.append(final_pose)
            processed_frames.append(image)
            trans_matrices.append(trans_matrix.copy())

        return np.array(processed_frames), np.array(normalized_frames), np.array(processed_poses), np.array(trans_matrices), np.array(bbox)


    def __getitem__(self, idx):
        item_path = self.items[idx]
        relative_path_split = item_path[len(self.root_dir):].split("/")
        action = relative_path_split[0]

        label = sio.loadmat(item_path + "/joint_positions")
        poses = label["pos_img"].T

        all_frames = sorted(glob.glob(item_path + "/*.png"))
        images = []
        for image_path in all_frames:
            images.append(io.imread(image_path))

        image_height = len(images[0])
        image_width = len(images[0][0])

        visibility = []
        for frame_pose in poses:
            frame_visibility = []
            for joint in frame_pose:
                x = joint[0]
                y = joint[1]

                if x < 0 or x > image_width or y < 0 or y > image_height:
                    frame_visibility.append(0)
                else:
                    frame_visibility.append(1)
            visibility.append(frame_visibility)

        processed_images, normalized_images, normalized_poses, trans_matrices, bbox = self.preprocess(np.array(images), np.array(poses))


        #action = label["action"][0]
        action_1h = np.zeros(21)
        action_1h[self.action_mapping[action]] = 1

        number_of_frames = len(normalized_images)
        if number_of_frames < 40:

            # padding for images
            desired_shape = normalized_images[0].shape
            blanks = np.zeros((40 - number_of_frames, desired_shape[0], desired_shape[1], desired_shape[2]))
            normalized_images = np.concatenate((normalized_images, blanks))

            # padding for poses
            assert normalized_poses.shape[1:] == (16, 3)
            blanks = np.zeros((40 - number_of_frames, 16, 3))
            normalized_poses = np.concatenate((normalized_poses, blanks))

            # padding for matrices
            assert trans_matrices.shape[1:] == (3, 3)
            blanks = np.zeros((40 - number_of_frames, 3, 3))
            trans_matrices = np.concatenate((trans_matrices, blanks))

        t_action_1h = torch.from_numpy(action_1h).float()
        t_normalized_frames = torch.from_numpy(normalized_images.reshape(-1, 3, self.final_size, self.final_size)).float()
        t_normalized_poses = torch.from_numpy(normalized_poses).float()
        t_trans_matrices = torch.from_numpy(trans_matrices).float()
        t_sequence_length = torch.from_numpy(np.array([number_of_frames])).int()
        t_bbox = torch.from_numpy(self.bbox).float()
        
        t_index = torch.zeros(1).float()
        t_index[0] = idx

        t_parameters = torch.zeros(5).float()
        t_parameters[0] = self.test["scale"]
        t_parameters[1] = float(self.test["angle"])
        t_parameters[2] = float(self.test["flip"])
        t_parameters[3] = float(self.test["trans_x"])
        t_parameters[4] = float(self.test["trans_y"])

        return {
            "action_1h": t_action_1h,
            "normalized_frames": t_normalized_frames,
            "normalized_poses": t_normalized_poses,
            "sequence_length": t_sequence_length,
            "trans_matrices": t_trans_matrices,
            "parameters": t_parameters,
            "index": t_index,
            "bbox": t_bbox
        }
