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

class PennActionFragmentsDataset(data.Dataset):
    def __init__(self, root_dir, transform=None, use_random_parameters=False, train=True, val=False, augmentation_amount=1, use_gt_bb=False):
        self.root_dir = root_dir
        self.padding_amount = 8

        self.train = train
        self.val = val
        self.use_gt_bb = use_gt_bb
        self.augmentation_amount = augmentation_amount

        self.use_random_parameters = use_random_parameters

        if self.train:
            if self.val:
                self.train_test_folder = "val/"
            else:
                self.train_test_folder = "train/"
        else:
            self.train_test_folder = "test/"

        self.set_prefix("")

        self.number_of_fragments = len(glob.glob("{}{}indices/{}*".format(self.root_dir, self.prefix, self.train_test_folder)))

    def set_prefix(self, path):
        self.prefix = path

        self.images_folder = self.root_dir + self.prefix + "images/"
        self.annotation_folder = self.root_dir + self.prefix + "annotations/"
        self.indices_folder = self.root_dir + self.prefix + "indices/"

    def __len__(self):
        return self.number_of_fragments

    def __getitem__(self, idx):
        if self.use_random_parameters:
            dice_roll = random.randint(0, self.augmentation_amount)
            if dice_roll == 0:
                self.set_prefix("")
            else:
                self.set_prefix("rand{}_".format(dice_roll))

        padded_indice = str(idx).zfill(self.padding_amount)

        t_indices = torch.load(self.indices_folder + self.train_test_folder + padded_indice + ".indices.pt")

        padded_filename = str(int(t_indices[-1].item())).zfill(8)

        t_action = torch.load(self.annotation_folder + padded_filename + ".action_1h.pt")
        if not self.use_gt_bb:
            t_frames = torch.load(self.images_folder + padded_filename + ".frames.pt")
            t_poses = torch.load(self.annotation_folder + padded_filename + ".poses.pt")
            t_matrices = torch.load(self.annotation_folder + padded_filename + ".matrices.pt")
            t_bbox = torch.load(self.annotation_folder + padded_filename + ".bbox.pt")
        else:
            t_frames = torch.load(self.images_folder + padded_filename + ".frames_gt_bb.pt")
            t_poses = torch.load(self.annotation_folder + padded_filename + ".poses_gt_bb.pt")
            t_matrices = torch.load(self.annotation_folder + padded_filename + ".matrices_gt_bb.pt")
            t_bbox = torch.load(self.annotation_folder + padded_filename + ".bbox_gt_bb.pt")

        t_parameters = torch.load(self.annotation_folder + padded_filename + ".parameters.pt")
        t_original_window_size = torch.load(self.annotation_folder + padded_filename + ".original_window_size.pt")

        start = int(t_indices[0].item())
        end = int(t_indices[1].item())

        t_frames = t_frames[start:end]
        t_poses = t_poses[start:end]
        t_matrices = t_matrices[start:end]
        t_bbox = t_bbox[start:end]
        t_original_window_size = t_original_window_size[start:end]

        t_frames = 2.0 * (t_frames.float() / 255.0) - 1.0
        t_poses = t_poses.float()
        t_poses[:, :, 0:2] = t_poses[:, :, 0:2] / 255.0

        return {
            "frames": t_frames,
            "poses": t_poses,
            "action_1h": t_action,
            "trans_matrices": t_matrices,
            "indices": t_indices,
            "bbox": t_bbox,
            "parameters": t_parameters,
            "original_window_size": t_original_window_size
        }
