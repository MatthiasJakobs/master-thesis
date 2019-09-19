from datasets.JHMDBDataset import JHMDBDataset
from datasets.PennActionDataset import PennActionDataset
from visualization import show_predictions_ontop, show_pose_on_image
import matplotlib.pyplot as plt

import torch
import shutil

import numpy as np

import random
import skimage.io as io
import os
import glob

def delete_and_create(root_dir, use_random=False, split=1):
        base_list = []

        if use_random:
            prefix = "rand_"
        else:
            prefix = ""

        base_list.append(prefix + "images")
        base_list.append(prefix + "annotations")

        if not use_random:
                base_list.append(prefix + "indices/val/" + str(split))
                base_list.append(prefix + "indices/test/" + str(split))
        
        base_list.append(prefix + "indices/train/" + str(split))

        for i in base_list:
                folder_path = root_dir + "{}".format(i)
                if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                        os.makedirs(folder_path)
                else:
                        os.makedirs(folder_path)

def create_fragments_pennaction(train, val):
        ds = PennActionDataset("/data/mjakobs/data/pennaction/", use_random_parameters=False, train=train)

        print("-" * 50)
        print("Train: {}, Val: {}".format(train, val))
        print("-" * 50)

        length = len(ds)
        current = 0

        all_indices = list(range(len(ds)))

        if train:
                random.seed(1)
                random.shuffle(all_indices)
                ten_percent = int(0.1 * len(ds))
                train_indices = all_indices[ten_percent:]
                val_indices = all_indices[:ten_percent]

                if val:
                        train_test_folder = "val"
                        all_indices = val_indices
                else:
                        train_test_folder = "train"
                        all_indices = train_indices

        else:
                train_test_folder = "test"

        for counter, idx in enumerate(all_indices):
                entry = ds[idx]
                frames = ((entry["normalized_frames"] + 1 )/ 2.0 * 255.0).int()
                poses = entry["normalized_poses"]
                actions = entry["action_1h"]
                matrices = entry["trans_matrices"]
                bbox = entry["bbox"]
                parameters = entry["parameters"]

                num_frames = 16

                num_frames_total = len(frames)

                if num_frames_total < 16:
                        print("less than 16 frames")
                        continue

                mini_batches = int(num_frames_total / num_frames) + 1
                padded_image = str(idx).zfill(8)

                original_image = ds.indices[idx]
                padded_original_image = str(original_image).zfill(4)

                root_dir = "/data/mjakobs/data/pennaction_fragments/"

                if not os.path.exists(root_dir + "images/" + padded_original_image + ".frames.pt"):
                        torch.save(frames, root_dir + "images/" + padded_original_image + ".frames.pt")
                if not os.path.exists(root_dir + "annotations/" + padded_original_image + ".action_1h.pt"):
                        torch.save(actions, root_dir + "annotations/" + padded_original_image + ".action_1h.pt")
                if not os.path.exists(root_dir + "annotations/" + padded_original_image + ".poses.pt"):
                        torch.save(poses, root_dir + "annotations/" + padded_original_image + ".poses.pt")
                if not os.path.exists(root_dir + "annotations/" + padded_original_image + ".matrices.pt"):
                        torch.save(matrices, root_dir + "annotations/" + padded_original_image + ".matrices.pt")
                if not os.path.exists(root_dir + "annotations/" + padded_original_image + ".bbox.pt"):
                        torch.save(bbox, root_dir + "annotations/" + padded_original_image + ".bbox.pt")
                if not os.path.exists(root_dir + "annotations/" + padded_original_image + ".parameters.pt"):
                        torch.save(parameters, root_dir + "annotations/" + padded_original_image + ".parameters.pt")
                indices = torch.zeros((num_frames_total - num_frames), 2)

                for i in range(num_frames_total - num_frames):
                        start = i
                        end = min(i + num_frames, num_frames_total)

                        padded = str(current).zfill(8)
                        current = current + 1

                        indices = torch.zeros(3)
                        indices[0] = start
                        indices[1] = end
                        indices[2] = original_image

                        torch.save(indices, root_dir + "indices/{}/{}.indices.pt".format(train_test_folder, padded))
                        assert indices.shape == (3,)

                print("{} - {}: {} / {}".format(train_test_folder, split, counter+1, len(all_indices)))

def create_fragments_jhmdb(train, val, split, use_random=False):
        ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=use_random, use_saved_tensors=True, train=train, split=split)

        print("-" * 50)
        print("Train: {}, Val: {}, Split: {}, Random: {}".format(train, val, split, use_random))
        print("-" * 50)

        length = len(ds)
        current = 0

        root_dir = "/data/mjakobs/data/jhmdb_fragments/"

        if use_random:
            prefix = "rand_"
        else:
            prefix = ""

        all_indices = list(range(len(ds)))

        if train:
                random.seed(1)
                random.shuffle(all_indices)
                ten_percent = int(0.1 * len(ds))
                train_indices = all_indices[ten_percent:]
                val_indices = all_indices[:ten_percent]

                if val:
                        train_test_folder = "val"
                        all_indices = val_indices
                else:
                        train_test_folder = "train"
                        all_indices = train_indices

        else:
                train_test_folder = "test"

        for counter, idx in enumerate(all_indices):
                entry = ds[idx]
                frames = entry["normalized_frames"]
                poses = entry["normalized_poses"]
                actions = entry["action_1h"]
                sequence_length = entry["sequence_length"]
                matrices = entry["trans_matrices"]
                bbox = entry["bbox"]
                index = entry["index"]
                parameters = entry["parameters"]

                assert len(frames) == 40
                assert len(poses) == 40

                num_frames = 16

                num_frames_total = sequence_length[0]

                if num_frames_total < 16:
                        print("less than 16 frames")
                        continue

                mini_batches = int(num_frames_total / num_frames) + 1
                padded_image = str(idx).zfill(8)

                original_image = ds.indices[idx]
                padded_original_image = str(original_image).zfill(8)

                if not os.path.exists(root_dir + prefix + "images/" + padded_original_image + ".frames.pt"):
                        torch.save(frames, root_dir + prefix + "images/" + padded_original_image + ".frames.pt")
                if not os.path.exists(root_dir + prefix + "annotations/" + padded_original_image + ".action_1h.pt"):
                        torch.save(actions, root_dir + prefix + "annotations/" + padded_original_image + ".action_1h.pt")
                if not os.path.exists(root_dir + prefix + "annotations/" + padded_original_image + ".poses.pt"):
                        torch.save(poses, root_dir + prefix + "annotations/" + padded_original_image + ".poses.pt")
                if not os.path.exists(root_dir + prefix + "annotations/" + padded_original_image + ".matrices.pt"):
                        torch.save(matrices, root_dir + prefix + "annotations/" + padded_original_image + ".matrices.pt")
                if not os.path.exists(root_dir + prefix + "annotations/" + padded_original_image + ".index.pt"):
                        torch.save(index, root_dir + prefix + "annotations/" + padded_original_image + ".index.pt")
                if not os.path.exists(root_dir + prefix + "annotations/" + padded_original_image + ".bbox.pt"):
                        torch.save(bbox, root_dir + prefix + "annotations/" + padded_original_image + ".bbox.pt")
                if not os.path.exists(root_dir + prefix + "annotations/" + padded_original_image + ".parameters.pt"):
                        torch.save(parameters, root_dir + prefix + "annotations/" + padded_original_image + ".parameters.pt")


                indices = torch.zeros((num_frames_total - num_frames), 2)

                for i in range(num_frames_total - num_frames):
                        start = i
                        end = min(i + num_frames, num_frames_total)

                        padded = str(current).zfill(8)
                        current = current + 1

                        indices = torch.zeros(3)
                        indices[0] = start
                        indices[1] = end
                        indices[2] = original_image

                        torch.save(indices, root_dir + prefix + "indices/{}/{}/{}.indices.pt".format(train_test_folder, str(split), padded))
                        assert indices.shape == (3,)

                print("{} - {}: {} / {}".format(train_test_folder, split, counter+1, len(all_indices)))

split = 1

# for use_random in [True, False]:
#         delete_and_create("/data/mjakobs/data/jhmdb_fragments/", use_random=use_random)
#         create_fragments_jhmdb(True, False, split, use_random=use_random)

# create_fragments_jhmdb(False, False, split)
# create_fragments_jhmdb(True, True, split)


delete_and_create("/data/mjakobs/data/pennaction_fragments/")
create_fragments_pennaction(False, False)
create_fragments_pennaction(True, True)
create_fragments_pennaction(True, False)