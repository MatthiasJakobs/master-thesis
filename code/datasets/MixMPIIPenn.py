import torch
import torch.utils.data as data

from datasets.PennActionDataset import PennActionDataset
from datasets.MPIIDataset import MPIIDataset
from datasets.BaseDataset import BaseDataset

import random
import math

import glob
import random

class MixMPIIPenn(BaseDataset):
    def __init__(self, transform=None, train=True, val=False, use_gt_bb=False):

        # Decision: Always using random parameters (if not val) and saved tensors

        super().__init__("irrelevant", train=train, val=val)

        assert train # should not be used for testing

        if val == True:
            use_random = False
        else:
            use_random = True

        self.pennaction = PennActionDataset("/data/mjakobs/data/pennaction/", use_random_parameters=use_random, train=train, val=val, use_gt_bb=True, use_saved_tensors=True, augmentation_amount=6)
        self.mpii = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=use_random, use_saved_tensors=True, train=train, val=val, augmentation_amount=3)

        self.padding_amount = 8

        self.sample_amount = 11 # this leads to a approximate 2/3 mpii 1/3 pennaction mix

        assert augmentation_amount > 0
        self.augmentation_amount = augmentation_amount

    def __len__(self):
        return len(self.mpii) + len(self.pennaction) * self.sample_amount

    def __getitem__(self, idx):

        if idx >= len(self.mpii):
            # pennaction
            real_index = math.floor((idx - len(self.mpii)) / self.sample_amount)

            entry = self.pennaction[real_index]
            clip_length = len(entry["normalized_frames"])
            frame_idx = random.randint(0, clip_length-1)

            image = entry["normalized_frames"][frame_idx]
            pose = entry["normalized_poses"][frame_idx]
            matrix = entry["trans_matrices"][frame_idx]
            bbox = entry["bbox"][frame_idx]
            dataset = "pennaction"
            
        else:
            # mpii
            entry = self.mpii[idx]
            image = entry["normalized_image"]
            pose = entry["normalized_pose"]
            matrix = entry["trans_matrix"]
            bbox = entry["bbox"]
            dataset = "mpii"

        return {
            "normalized_image": image,
            "normalized_pose": pose,
            "trans_matrix": matrix,
            "bbox": bbox,
            "dataset": dataset
        }

