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

def create_folder_if_not_present(filepath):
        if not os.path.exists(filepath):
                os.makedirs(filepath)       

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

def create_fragments_pennaction(train=False, val=False, use_random=False, subprefix="1"):
        ds = PennActionDataset("/data/mjakobs/data/pennaction/", use_random_parameters=use_random, train=train, val=val)
        ds_bbox_gt = PennActionDataset("/data/mjakobs/data/pennaction/", use_random_parameters=use_random, train=train, val=val, use_gt_bb=True)

        print("-" * 50)
        print("Train: {}, Val: {}".format(train, val))
        print("-" * 50)

        length = len(ds)
        current = 0

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
                matrices = entry["trans_matrices"]
                bbox = entry["bbox"]
                parameters = entry["parameters"]
                original_window_size = entry["original_window_size"]

                frames = ((frames + 1) / 2.0) * 255.0
                frames = frames.byte()

                poses[:, :, 0:2] = poses[:, :, 0:2] * 255.0
                poses = poses.int()

                root_dir = "/data/mjakobs/data/pennaction/"
                original_image = ds.indices[idx]
                padded_original_image = str(original_image).zfill(8)

                torch.save(frames, root_dir + prefix + "images/" + padded_original_image + ".frames.pt")
                torch.save(actions, root_dir + prefix + "annotations/" + padded_original_image + ".action_1h.pt")
                torch.save(poses, root_dir + prefix + "annotations/" + padded_original_image + ".poses.pt")
                torch.save(matrices, root_dir + prefix + "annotations/" + padded_original_image + ".matrices.pt")
                torch.save(bbox, root_dir + prefix + "annotations/" + padded_original_image + ".bbox.pt")
                torch.save(parameters, root_dir + prefix + "annotations/" + padded_original_image + ".parameters.pt")
                torch.save(original_window_size, root_dir + prefix + "annotations/" + padded_original_image + ".original_window_size.pt")

                entry = ds_bbox_gt[idx]
                frames_gt_bb = entry["normalized_frames"]
                poses_gt_bb = entry["normalized_poses"]
                matrices_gt_bb = entry["trans_matrices"]
                bbox_gt_bb = entry["bbox"]

                frames_gt_bb = ((frames_gt_bb + 1) / 2.0) * 255.0
                frames_gt_bb = frames_gt_bb.byte()

                poses_gt_bb[:, :, 0:2] = poses_gt_bb[:, :, 0:2] * 255.0
                poses_gt_bb = poses_gt_bb.int()

                torch.save(frames_gt_bb, root_dir + prefix + "images/" + padded_original_image + ".frames_gt_bb.pt")
                torch.save(poses_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".poses_gt_bb.pt")
                torch.save(matrices_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".matrices_gt_bb.pt")
                torch.save(bbox_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".bbox_gt_bb.pt")

                num_frames = 16

                num_frames_total = len(frames)

                if num_frames_total < 16:
                        print("less than 16 frames")
                        continue

                root_dir = "/data/mjakobs/data/pennaction_fragments/"

                torch.save(frames, root_dir + prefix + "images/" + padded_original_image + ".frames.pt")
                torch.save(actions, root_dir + prefix + "annotations/" + padded_original_image + ".action_1h.pt")
                torch.save(poses, root_dir + prefix + "annotations/" + padded_original_image + ".poses.pt")
                torch.save(matrices, root_dir + prefix + "annotations/" + padded_original_image + ".matrices.pt")
                torch.save(bbox, root_dir + prefix + "annotations/" + padded_original_image + ".bbox.pt")
                torch.save(parameters, root_dir + prefix + "annotations/" + padded_original_image + ".parameters.pt")
                torch.save(original_window_size, root_dir + prefix + "annotations/" + padded_original_image + ".original_window_size.pt")

                torch.save(frames_gt_bb, root_dir + prefix + "images/" + padded_original_image + ".frames_gt_bb.pt")
                torch.save(poses_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".poses_gt_bb.pt")
                torch.save(matrices_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".matrices_gt_bb.pt")
                torch.save(bbox_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".bbox_gt_bb.pt")
                
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

                        torch.save(indices, root_dir + prefix + "indices/{}/{}.indices.pt".format(train_test_folder, padded))
                        assert indices.shape == (3,)

                print("{}: {} / {}".format(train_test_folder, counter+1, len(all_indices)))

def create_fragments_jhmdb(train=False, val=False, split=1, use_random=False, subprefix="1"):
        ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=use_random, use_saved_tensors=False, train=train, val=val, split=split)
        ds_bbox_gt = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=use_random, use_saved_tensors=False, train=train, val=val, split=split, use_gt_bb=True)

        if use_random:
            assert train

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
                original_window_size = entry["original_window_size"]

                frames = ((frames + 1) / 2.0) * 255.0
                frames = frames.byte()

                poses[:, :, 0:2] = poses[:, :, 0:2] * 255.0
                poses = poses.int()

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
                torch.save(original_window_size, root_dir + prefix + "annotations/" + padded_original_image + ".original_window_size.pt")

                entry = ds_bbox_gt[idx]
                frames_gt_bb = entry["normalized_frames"]
                poses_gt_bb = entry["normalized_poses"]
                matrices_gt_bb = entry["trans_matrices"]
                bbox_gt_bb = entry["bbox"]

                frames_gt_bb = ((frames_gt_bb + 1) / 2.0) * 255.0
                frames_gt_bb = frames_gt_bb.byte()

                poses_gt_bb[:, :, 0:2] = poses_gt_bb[:, :, 0:2] * 255.0
                poses_gt_bb = poses_gt_bb.int()

                torch.save(frames_gt_bb, root_dir + prefix + "images/" + padded_original_image + ".frames_gt_bb.pt")
                torch.save(poses_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".poses_gt_bb.pt")
                torch.save(matrices_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".matrices_gt_bb.pt")
                torch.save(bbox_gt_bb, root_dir + prefix + "annotations/" + padded_original_image + ".bbox_gt_bb.pt")


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

def create_fragments_mpii(train=False, val=False, use_random=False, subprefix="1", split=1):
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
                image_path = entry["image_path"]

                frame = ((frame + 1) / 2.0) * 255.0
                frame = frame.byte()

                pose[:, 0:2] = pose[:, 0:2] * 255.0
                pose = pose.int()

                original_image = ds.indices[idx]
                padded_original_image = str(original_image).zfill(8)

                torch.save(frame, root_dir + train_test_folder + prefix + "images/" + padded_original_image + ".frame.pt")
                torch.save(headsize, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".headsize.pt")
                torch.save(pose, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".pose.pt")
                torch.save(matrix, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".matrix.pt")
                torch.save(bbox, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".bbox.pt")
                torch.save(parameters, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".parameters.pt")
                torch.save(image_path, root_dir + train_test_folder + prefix + "annotations/" + padded_original_image + ".image_path.pt")

def complete_jhmdb(split=1, amount_random=6):
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

def complete_mpii():
        ######
        #  MPII
        #####
        create_folder_if_not_present("/data/mjakobs/data/mpii/train/annotations")
        create_folder_if_not_present("/data/mjakobs/data/mpii/train/images")
        
        create_folder_if_not_present("/data/mjakobs/data/mpii/train/rand1_annotations")
        create_folder_if_not_present("/data/mjakobs/data/mpii/train/rand1_images")
        create_folder_if_not_present("/data/mjakobs/data/mpii/train/rand2_annotations")
        create_folder_if_not_present("/data/mjakobs/data/mpii/train/rand2_images")
        create_folder_if_not_present("/data/mjakobs/data/mpii/train/rand3_annotations")
        create_folder_if_not_present("/data/mjakobs/data/mpii/train/rand3_images")
        
        create_folder_if_not_present("/data/mjakobs/data/mpii/val/annotations")
        create_folder_if_not_present("/data/mjakobs/data/mpii/val/images")

        create_fragments_mpii(train=True, val=False, use_random=False) # Train no random
        create_fragments_mpii(train=True, val=False, use_random=True, subprefix="1") # train random 1
        create_fragments_mpii(train=True, val=False, use_random=True, subprefix="2") # train random 2
        create_fragments_mpii(train=True, val=False, use_random=True, subprefix="3") # train random 3
        create_fragments_mpii(train=True, val=True, use_random=False) # val

def complete_pennaction():
        ######
        #  Penn Action
        # #####
        create_folder_if_not_present("/data/mjakobs/data/pennaction/annotations")
        create_folder_if_not_present("/data/mjakobs/data/pennaction/images")
        create_folder_if_not_present("/data/mjakobs/data/pennaction/rand1_annotations")
        create_folder_if_not_present("/data/mjakobs/data/pennaction/rand1_images")
        create_folder_if_not_present("/data/mjakobs/data/pennaction/rand2_annotations")
        create_folder_if_not_present("/data/mjakobs/data/pennaction/rand2_images")
        create_folder_if_not_present("/data/mjakobs/data/pennaction/rand3_annotations")
        create_folder_if_not_present("/data/mjakobs/data/pennaction/rand3_images")

        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/annotations")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/images")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand1_indices")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand1_indices/train")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand1_annotations")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand1_images")
        
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand2_indices")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand2_indices/train")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand2_annotations")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand2_images")
        
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand3_indices")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand3_indices/train")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand3_annotations")
        create_folder_if_not_present("/data/mjakobs/data/pennaction_fragments/rand3_images")

        create_fragments_pennaction(train=False, val=False) # test
        create_fragments_pennaction(train=True, val=True) # val
        create_fragments_pennaction(train=True, val=False, use_random=False) # train, no random
        create_fragments_pennaction(train=True, val=False, use_random=True, subprefix="1") # train 1, random
        create_fragments_pennaction(train=True, val=False, use_random=True, subprefix="2") # train 2, random
        create_fragments_pennaction(train=True, val=False, use_random=True, subprefix="3") # train 3, random

def complete_recreation():
        complete_jhmdb()
        complete_mpii()
        complete_pennaction()
