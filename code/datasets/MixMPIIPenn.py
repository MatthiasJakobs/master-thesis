import torch
import torch.utils.data as data

from datasets.PennActionDataset import PennActionDataset
from datasets.MPIIDataset import MPIIDataset

import glob
import random

class MixMPIIPenn(data.Dataset):
    def __init__(self, transform=None, use_random_parameters=False, train=True, val=False, augmentation_amount=1):

        assert train # should not be used for testing

        self.pennaction = PennActionDataset("/data/mjakobs/data/pennaction/", use_random_parameters=use_random_parameters, train=train, val=val, use_gt_bb=True)
        self.mpii = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=use_random_parameters, use_saved_tensors=True, train=train, val=val)

        self.padding_amount = 8

        self.sample_amount = 11 # this leads to a approximate 2/3 mpii 1/3 pennaction mix

        self.train = train
        self.val = val

        assert augmentation_amount > 0
        self.augmentation_amount = augmentation_amount

        self.mpii_count = 0
        self.pennaction_count = 0

    def __len__(self):
        return len(self.mpii) + len(self.pennaction) * self.sample_amount

    def __getitem__(self, idx):

        if idx > len(self.mpii):
            # pennaction
            self.pennaction_count = self.pennaction_count + 1
            real_index = idx / self.sample_amount
            # entry = self.pennaction[real_index]
            # clip_length = len(entry["normalized_frames"])
            # frame_idx = random.randint(0, clip_length)

            # return {
            #     "normalized_image": entry["normalized_frames"][frame_idx],
            #     "normalized_pose": entry["normalized_poses"][frame_idx],
            #     "trans_matrix": entry["trans_matrices"][frame_idx],
            #     "bbox": entry["bbox"][frame_idx]
            # }
            return {}
        else:
            self.mpii_count = self.mpii_count + 1
            # mpii
            #return self.mpii[idx]
            return {}