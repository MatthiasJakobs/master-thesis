import torch
import torch.utils.data as data

import glob

class JHMDBFragmentsDataset(data.Dataset):
    def __init__(self, root_dir, transform=None, use_random_parameters=False, split=1, train=True, val=False):
        self.root_dir = root_dir
        self.padding_amount = 8

        self.split = split
        self.train = train
        self.val = val

        self.use_random_parameters = use_random_parameters

        if self.use_random_parameters:
            self.prefix = "rand_"
        else:
            self.prefix = ""

        self.images_folder = self.root_dir + self.prefix + "images/"
        self.annotation_folder = self.root_dir + self.prefix + "annotations/"
        self.indices_folder = self.root_dir + self.prefix + "indices/"

        if self.train:
            if self.val:
                self.train_test_folder = "val/"
            else:
                self.train_test_folder = "train/"
        else:
            self.train_test_folder = "test/"

        self.number_of_fragments = len(glob.glob("{}{}indices/{}{}/*".format(self.root_dir, self.prefix, self.train_test_folder, str(self.split))))

    def __len__(self):
        return self.number_of_fragments

    def __getitem__(self, idx):
        padded_indice = str(idx).zfill(self.padding_amount)
        
        t_indices = torch.load(self.indices_folder + self.train_test_folder + str(self.split) + "/" + padded_indice + ".indices.pt")

        padded_filename = str(int(t_indices[-1].item())).zfill(self.padding_amount)

        t_poses = torch.load(self.annotation_folder + padded_filename + ".poses.pt")
        t_action = torch.load(self.annotation_folder + padded_filename + ".action_1h.pt")
        t_frames = torch.load(self.images_folder + padded_filename + ".frames.pt")
        t_matrices = torch.load(self.annotation_folder + padded_filename + ".matrices.pt")

        start = int(t_indices[0].item())
        end = int(t_indices[1].item())

        t_frames = t_frames[start:end]
        t_poses = t_poses[start:end]
        t_matrices = t_matrices[start:end]

        return {
            "frames": t_frames,
            "poses": t_poses,
            "action_1h": t_action,
            "trans_matrices": t_matrices
        }
