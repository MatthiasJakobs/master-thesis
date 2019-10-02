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
from deephar.utils import transform_2d_point, translate, scale, flip_h, superflatten, transform_pose, get_valid_joints, flip_lr_pose


class BaseDataset(data.Dataset):

    ### Definitions:
    # self.indices contains the indices of all items matching the current configuration, i.e., all training clips for training 
    # on video or all validation images for validating on an image dataset.
    #
    # self.items contains __all__ items belonging to a dataset, including test. This way, the differentiation
    # between train, test and val is only in what indices are used

    def __init__(self, root_dir, transform=None, use_random_parameters=False, use_saved_tensors=False, train=True, val=False):
        self.root_dir = root_dir
        self.train = train
        self.val = val

        if not train:
            assert not use_random_parameters, print("Test files cannot be augmented")
        
        if train and val:
            assert not use_random_parameters, print("Validation files cannot be augmented")

        self.use_random_parameters = use_random_parameters
        self.use_saved_tensors = use_saved_tensors

        self.final_size = 255

        self.val_split_amount = 0.1

        self.indices = []
        self.items = []

        self.angles=torch.ByteTensor([0, 0, 0])
        self.scales=torch.FloatTensor([1., 1., 1.])
        self.flip_horizontal = torch.ByteTensor([0, 0])
        self.trans_x=torch.FloatTensor([0., 0., 0.])
        self.trans_y=torch.FloatTensor([0., 0., 0.])
        self.subsampling=[1, 1]

    def __len__(self):
        return len(self.indices)

    def set_augmentation_parameters(self):
        self.aug_conf = {}
        self.aug_conf["scale"] = self.scales[np.random.randint(0, len(self.scales))]
        self.aug_conf["angle"] = self.angles[np.random.randint(0, len(self.angles))]
        self.aug_conf["flip"] = self.flip_horizontal[np.random.randint(0, len(self.flip_horizontal))]
        self.aug_conf["trans_x"] = self.trans_x[np.random.randint(0, len(self.trans_x))]
        self.aug_conf["trans_y"] = self.trans_y[np.random.randint(0, len(self.trans_y))]


    def calc_bbox_and_center(self, width, height, pre_bb=None, offset=None):
        window_size =  self.aug_conf["scale"] * max(height, width)

        if pre_bb is not None:
            bbox = pre_bb
        else:
            bbox = torch.IntTensor([
                int(width / 2) - (window_size / 2), # x1, upper left
                int(height / 2) - (window_size / 2), # y1, upper left
                int(width / 2) + (window_size / 2), # x2, lower right
                int(height / 2) + (window_size / 2)  # y2, lower right
            ])

        self.bbox = bbox

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

        #assert bbox_width >= 32 and bbox_height >= 32

        center += torch.IntTensor(self.aug_conf["scale"] * torch.IntTensor([self.aug_conf["trans_x"], self.aug_conf["trans_y"]]))

        self.center = center

    def set_visibility(self, pose):
        valid_joints = get_valid_joints(pose, need_sum=False)[:, 0:2]
        visibility = torch.from_numpy(np.apply_along_axis(np.all, 1, valid_joints).astype(np.uint8)).float()
        pose[:, 2] = visibility
        return pose

    def preprocess(self, frame, pose):
        trans_matrix, image = rotate_and_crop(frame, self.aug_conf["angle"], self.center, self.window_size)
        size_after_rotate = torch.FloatTensor([image.shape[1], image.shape[0]])

        image = resize(image, (self.final_size, self.final_size), preserve_range=True)
        trans_matrix = scale(trans_matrix, self.final_size / size_after_rotate[0], self.final_size / size_after_rotate[1])

        # randomly flip horizontal
        if self.aug_conf["flip"]:
            image = np.fliplr(image)

            trans_matrix = translate(trans_matrix, -image.shape[1] / 2, -image.shape[0] / 2)
            trans_matrix = flip_h(trans_matrix)
            trans_matrix = translate(trans_matrix, image.shape[1] / 2, image.shape[0] / 2)

        # normalize pose to [0,1] and image to [-1, 1]
        normalized_image = normalize_channels(image)
        trans_matrix = scale(trans_matrix, 1.0 / self.final_size, 1.0 / self.final_size)
        transformed_pose = transform_pose(trans_matrix, pose)

        trans_matrix = torch.from_numpy(trans_matrix).float()
        normalized_image = torch.from_numpy(normalized_image).float()
        transformed_pose = torch.from_numpy(transformed_pose).float()

        assert normalized_image.max() <= 1 and normalized_image.min() >= -1

        return trans_matrix, normalized_image, transformed_pose


    def __getitem__(self, idx):
        return {}