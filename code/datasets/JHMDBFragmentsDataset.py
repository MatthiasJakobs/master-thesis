import torch
import torch.utils.data as data

import glob

class JHMDBFragmentsDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.padding_amount = 8

        self.number_of_fragments = len(glob.glob(self.root_dir + "images/*"))

    def __len__(self):
        return self.number_of_fragments

    def __getitem__(self, idx):
        self.images_folder = self.root_dir + "images/"
        self.annotation_folder = self.root_dir + "annotations/"

        padded = str(idx).zfill(self.padding_amount)

        t_frames = torch.load(self.images_folder + padded + ".frames.pt")
        t_poses = torch.load(self.images_folder + padded + ".poses.pt")
        t_action = torch.load(self.images_folder + padded + ".action_1h.pt")

        return {
            "frames": t_frames,
            "poses": t_poses,
            "action": t_action
        }