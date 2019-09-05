import torch
import torch.utils.data as data

import glob

class JHMDBFragmentsDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.padding_amount = 8

        self.images_folder = self.root_dir + "images/"
        self.annotation_folder = self.root_dir + "annotations/"
        self.indices_folder = self.root_dir + "indices/"

        self.number_of_fragments = len(glob.glob(self.root_dir + "indices/*"))

    def __len__(self):
        return self.number_of_fragments

    def __getitem__(self, idx):
        padded_indice = str(idx).zfill(self.padding_amount)

        t_indices = torch.load(self.indices_folder + padded_indice + ".indices.pt")

        padded_filename = str(t_indices[-1]).zfill(self.padding_amount)

        t_poses = torch.load(self.annotation_folder + padded_filename + ".poses.pt")
        t_action = torch.load(self.annotation_folder + padded_filename + ".action_1h.pt")
        t_frames = torch.load(self.images_folder + padded_filename + ".frames.pt")

        start = t_indices[0]
        end = t_indices[1]

        t_frames = t_frames[start:end]

        return {
            "frames": t_frames,
            "poses": t_poses,
            "action": t_action
        }