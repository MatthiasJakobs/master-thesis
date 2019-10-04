from datasets.JHMDBDataset import JHMDBDataset
from datasets.PennActionDataset import PennActionDataset
from datasets.MPIIDataset import MPIIDataset
from visualization import show_predictions_ontop, show_pose_on_image
import matplotlib.pyplot as plt

import torch
import shutil

import numpy as np

import random
import skimage.io as io
import os
import glob

def delete_and_create(root_dir, use_random=False, split=1, subprefix="2"):
        base_list = []

        if use_random:
            prefix = "rand{}_".format(subprefix)
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

def create_fragments_jhmdb(train=False, val=False, split=1, use_random=False, subprefix="1"):
        ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=use_random, use_saved_tensors=False, train=train, val=val, split=split)

        print("-" * 50)
        print("Train: {}, Val: {}, Split: {}, Random: {}".format(train, val, split, use_random))
        print("-" * 50)

        length = len(ds)
        current = 0

        root_dir = "/data/mjakobs/data/jhmdb_fragments/"

        if use_random:
            prefix = "rand{}_".format(subprefix)
        else:
            prefix = ""

        all_indices = list(range(len(ds)))

        if train:
                if val:
                        train_test_folder = "val"
                else:
                        train_test_folder = "train"

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

                original_image = ds.indices[idx]
                padded_original_image = str(original_image).zfill(8)

                torch.save(frames, root_dir + prefix + "images/" + padded_original_image + ".frames.pt")
                torch.save(actions, root_dir + prefix + "annotations/" + padded_original_image + ".action_1h.pt")
                torch.save(poses, root_dir + prefix + "annotations/" + padded_original_image + ".poses.pt")
                torch.save(matrices, root_dir + prefix + "annotations/" + padded_original_image + ".matrices.pt")
                torch.save(index, root_dir + prefix + "annotations/" + padded_original_image + ".index.pt")
                torch.save(bbox, root_dir + prefix + "annotations/" + padded_original_image + ".bbox.pt")
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

def create_fragments_mpii(train=False, val=False, use_random=False, subprefix="1"):
        ds = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=use_random, use_saved_tensors=False, train=train, val=val)

        print("-" * 50)
        print("Train: {}, Val: {}, Split: {}, Random: {}".format(train, val, split, use_random))
        print("-" * 50)

        length = len(ds)
        current = 0

        assert train == True

        root_dir = "/data/mjakobs/data/mpii/"

        if use_random:
            prefix = "rand{}_".format(subprefix)
        else:
            prefix = ""

        all_indices = list(range(len(ds)))

        if val:
                train_test_folder = "val/"
        else:
                train_test_folder = "train/"


        for counter, idx in enumerate(all_indices):
                entry = ds[idx]
                frame = entry["normalized_image"]
                pose = entry["normalized_pose"]
                matrix = entry["trans_matrix"]
                bbox = entry["bbox"]
                headsize = entry["head_size"]
                parameters = entry["parameters"]

                original_image = ds.indices[idx]
                padded_original_image = str(original_image).zfill(8)

                #if not os.path.exists(root_dir + train_test_folder + prefix + "images/" + padded_original_image + ".frame.pt"):
                torch.save(frame, root_dir + train_test_folder + prefix + "images/" + padded_original_image + ".frame.pt")
                #if not os.path.exists(root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".headsize.pt"):
                torch.save(headsize, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".headsize.pt")
                #if not os.path.exists(root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".pose.pt"):
                torch.save(pose, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".pose.pt")
                #if not os.path.exists(root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".matrix.pt"):
                torch.save(matrix, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".matrix.pt")
                #if not os.path.exists(root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".bbox.pt"):
                torch.save(bbox, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".bbox.pt")
                #if not os.path.exists(root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".parameters.pt"):
                torch.save(parameters, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".parameters.pt")


split = 1
amount_random = 1

######
#  JHMDB
#####
for i in range(amount_random):
        subprefix = "{}".format(i + 1)

        delete_and_create("/data/mjakobs/data/jhmdb_fragments/", use_random=True, subprefix=subprefix)
        create_fragments_jhmdb(train=True, val=False, split=split, use_random=True, subprefix=subprefix)

delete_and_create("/data/mjakobs/data/jhmdb_fragments/", use_random=False, subprefix="1")
create_fragments_jhmdb(train=True, val=False, split=split, use_random=False)
create_fragments_jhmdb(train=False, val=False, split=split) # test
create_fragments_jhmdb(train=True, val=True, split=split) # val

######
#  MPII
#####
create_fragments_mpii(train=True, val=False, use_random=False) # Train nrandom
create_fragments_mpii(train=True, val=False, use_random=True) # train random
create_fragments_mpii(train=True, val=True, use_random=False) # val

######
#  Penn Action
#####
delete_and_create("/data/mjakobs/data/pennaction_fragments/")
create_fragments_pennaction(False, False)
create_fragments_pennaction(True, True)
create_fragments_pennaction(True, False)