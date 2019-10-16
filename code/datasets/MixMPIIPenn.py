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
    def __init__(self, root_dir, transform=None, use_random_parameters=False, use_saved_tensors=False, train=True, val=False, augmentation_amount=1):

        super().__init__(root_dir, use_random_parameters=use_random_parameters, use_saved_tensors=use_saved_tensors, train=train, val=val)

        assert train # should not be used for testing

        self.pennaction = PennActionDataset("/data/mjakobs/data/pennaction/", use_random_parameters=use_random_parameters, train=train, val=val, use_gt_bb=True)
        self.mpii = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=use_random_parameters, use_saved_tensors=False, train=train, val=val)

        self.padding_amount = 8

        self.sample_amount = 11 # this leads to a approximate 2/3 mpii 1/3 pennaction mix

        assert augmentation_amount > 0
        self.augmentation_amount = augmentation_amount

    def __len__(self):
        return len(self.mpii) + len(self.pennaction) * self.sample_amount

    def __getitem__(self, idx):

        if self.val:
            train_test_folder = "val/"
        else:
            train_test_folder = "train/"

        if not self.val and self.use_random_parameters:
            dice_roll = random.randint(0, self.augmentation_amount)
            if dice_roll == 0:
                self.mpii.skip_random = True
                self.pennaction.skip_random = True
            else:
                self.mpii.skip_random = False
                self.pennaction.skip_random = False
                train_test_folder = "rand_train/"

        name_path = self.root_dir + train_test_folder
        item_name = str(idx).zfill(self.padding_amount)

        if idx > len(self.mpii):
            # pennaction
            real_index = math.floor((idx - len(self.mpii)) / self.sample_amount)
            if self.use_saved_tensors:
                image = torch.load(name_path + "images/" + item_name + ".image.pt")
                pose = torch.load(name_path + "annotations/" + item_name + ".pose.pt")
                matrix = torch.load(name_path + "annotations/" + item_name + ".matrix.pt")
                bbox = torch.load(name_path + "annotations/" + item_name + ".bbox.pt")
            else:
                entry = self.pennaction[real_index]
                clip_length = len(entry["normalized_frames"])
                frame_idx = random.randint(0, clip_length)

                image = entry["normalized_frames"][frame_idx]
                pose = entry["normalized_poses"][frame_idx]
                matrix = entry["trans_matrices"][frame_idx]
                bbox = entry["bbox"][frame_idx]
                
                torch.save(image, name_path + "images/" + item_name + ".image.pt")
                torch.save(pose, name_path + "annotations/" + item_name + ".pose.pt")
                torch.save(matrix, name_path + "annotations/" + item_name + ".matrix.pt")
                torch.save(bbox, name_path + "annotations/" + item_name + ".bbox.pt")
        else:
            # mpii
            self.mpii.skip_random = True

            if self.use_saved_tensors:
                image = torch.load(name_path + "images/" + item_name + ".image.pt")
                pose = torch.load(name_path + "annotations/" + item_name + ".pose.pt")
                matrix = torch.load(name_path + "annotations/" + item_name + ".matrix.pt")
                bbox = torch.load(name_path + "annotations/" + item_name + ".bbox.pt")                
            else:
                entry = self.mpii[idx]
                image = entry["normalized_image"]
                pose = entry["normalized_pose"]
                matrix = entry["trans_matrix"]
                bbox = entry["bbox"]
                
                torch.save(image, name_path + "images/" + item_name + ".image.pt")
                torch.save(pose, name_path + "annotations/" + item_name + ".pose.pt")
                torch.save(matrix, name_path + "annotations/" + item_name + ".matrix.pt")
                torch.save(bbox, name_path + "annotations/" + item_name + ".bbox.pt")

        return {
            "normalized_image": image,
            "normalized_pose": pose,
            "trans_matrix": matrix,
            "bbox": bbox
        }

