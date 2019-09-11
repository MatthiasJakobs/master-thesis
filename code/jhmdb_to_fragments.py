from datasets.JHMDBDataset import JHMDBDataset
from visualization import show_predictions_ontop, show_pose_on_image
import matplotlib.pyplot as plt

import torch
import shutil

import skimage.io as io
import os
import glob

for i in ["images", "annotations", "indices/train/1", "indices/train/2", "indices/train/3", "indices/test/1", "indices/test/2", "indices/test/3"]:
        folder_path = "/data/mjakobs/data/jhmdb_fragments/{}".format(i)
        if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                os.makedirs(folder_path)
        else:
                os.makedirs(folder_path)


train = True
split = 1

ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=False, use_saved_tensors=True, train=train, split=split)

print("-" * 50)
print("Train: {}, Split: {}".format(train, split))
print("-" * 50)

length = len(ds)
current = 0

if train:
        train_test_folder = "train"
else:
        train_test_folder = "test"

for idx, entry in enumerate(ds):
        frames = entry["normalized_frames"]
        poses = entry["normalized_poses"]
        actions = entry["action_1h"]
        sequence_length = entry["sequence_length"]
        matrices = entry["trans_matrices"]

        assert len(frames) == 40
        assert len(poses) == 40

        num_frames = 16

        num_frames_total = sequence_length[0]

        if num_frames_total < 16:
                print("less than 16 frames")
                continue

        mini_batches = int(num_frames_total / num_frames) + 1
        padded_image = str(idx).zfill(8)

        if not os.path.exists("/data/mjakobs/data/jhmdb_fragments/images/" + padded_image + ".frames.pt"):
                torch.save(frames, "/data/mjakobs/data/jhmdb_fragments/images/" + padded_image + ".frames.pt")
        if not os.path.exists("/data/mjakobs/data/jhmdb_fragments/annotations/" + padded_image + ".action_1h.pt"):
                torch.save(actions, "/data/mjakobs/data/jhmdb_fragments/annotations/" + padded_image + ".action_1h.pt")
        if not os.path.exists("/data/mjakobs/data/jhmdb_fragments/annotations/" + padded_image + ".poses.pt"):
                torch.save(poses, "/data/mjakobs/data/jhmdb_fragments/annotations/" + padded_image + ".poses.pt")
        if not os.path.exists("/data/mjakobs/data/jhmdb_fragments/annotations/" + padded_image + ".matrices.pt"):
                torch.save(matrices, "/data/mjakobs/data/jhmdb_fragments/annotations/" + padded_image + ".matrices.pt")

        indices = torch.zeros((num_frames_total - num_frames), 2)

        for i in range(num_frames_total - num_frames):
                start = i
                end = min(i + num_frames, num_frames_total)

                padded = str(current).zfill(8)
                current = current + 1

                indices = torch.zeros(3)
                indices[0] = start
                indices[1] = end
                indices[2] = idx

                torch.save(indices, "/data/mjakobs/data/jhmdb_fragments/indices/{}/{}/{}.indices.pt".format(train_test_folder, str(split), padded))
                assert indices.shape == (3,)

        print("{} - {}: {} / {}".format(train_test_folder, split, idx, length))
