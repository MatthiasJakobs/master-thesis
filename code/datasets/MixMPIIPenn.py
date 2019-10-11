import torch
import torch.utils.data as data

from datasets.PennActionFragmentsDataset import PennActionFragmentsDataset
from datasets.MPIIDataset import MPIIDataset

import glob
import random

class MixMPIIPenn(data.Dataset):
    def __init__(self, transform=None, use_random_parameters=False, train=True, val=False, augmentation_amount=1):

        assert train

        self.pennaction = PennActionFragmentsDataset("/data/mjakobs/data/pennaction_fragments/", use_random_parameters=use_random_parameters, train=train, val=val)
        self.mpii = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=use_random_parameters, use_saved_tensors=True, train=train, val=val)

        self.padding_amount = 8

        self.train = train
        self.val = val

        assert augmentation_amount > 0
        self.augmentation_amount = augmentation_amount

    def __len__(self):
        return len(self.pennaction) + len(self.mpii)

    def __getitem__(self, idx):
        dice_roll_dataset = random.randint(0, 3)
        # if dice_roll == 0 or 1 or 2 then use MPII, else PennAction


        if self.use_random_parameters:
            dice_roll = random.randint(0, self.augmentation_amount)
            if dice_roll == 0:
                self.set_prefix("")
            else:
                self.set_prefix("rand{}_".format(dice_roll))

        padded_indice = str(idx).zfill(self.padding_amount)

        t_indices = torch.load(self.indices_folder + self.train_test_folder + str(self.split) + "/" + padded_indice + ".indices.pt")

        padded_filename = str(int(t_indices[-1].item())).zfill(self.padding_amount)

        t_poses = torch.load(self.annotation_folder + padded_filename + ".poses.pt")
        t_action = torch.load(self.annotation_folder + padded_filename + ".action_1h.pt")
        t_bbox = torch.load(self.annotation_folder + padded_filename + ".bbox.pt")
        t_index = torch.load(self.annotation_folder + padded_filename + ".index.pt")
        t_parameters = torch.load(self.annotation_folder + padded_filename + ".parameters.pt")
        t_frames = torch.load(self.images_folder + padded_filename + ".frames.pt")
        t_matrices = torch.load(self.annotation_folder + padded_filename + ".matrices.pt")

        start = int(t_indices[0].item())
        end = int(t_indices[1].item())

        t_frames = t_frames[start:end]
        t_poses = t_poses[start:end]
        t_matrices = t_matrices[start:end]
        t_bboxes = t_bbox[start:end]

        return {
            "frames": t_frames,
            "poses": t_poses,
            "action_1h": t_action,
            "trans_matrices": t_matrices,
            "indices": t_indices,
            "bbox": t_bboxes,
            "parameters": t_parameters
        }
