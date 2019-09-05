from datasets.JHMDBDataset import JHMDBDataset
from visualization import show_predictions_ontop, show_pose_on_image
import matplotlib.pyplot as plt

import torch

import skimage.io as io

ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=False, use_saved_tensors=True)

length = len(ds)
current = 0
for idx, entry in enumerate(ds):
        frames = entry["normalized_frames"]
        poses = entry["normalized_poses"]
        actions = entry["action_1h"]
        sequence_length = entry["sequence_length"]

        num_frames = 16

        num_frames_total = sequence_length[0]
        if num_frames_total < 16:
                print("less than 16 frames")
                continue
        mini_batches = int(num_frames_total / num_frames) + 1
        padded_image = str(idx).zfill(8)

        torch.save(frames, "/data/mjakobs/data/jhmdb_fragments/images/" + padded_image + ".frames.pt")
        torch.save(actions, "/data/mjakobs/data/jhmdb_fragments/annotations/" + padded_image + ".action_1h.pt")
        torch.save(poses, "/data/mjakobs/data/jhmdb_fragments/annotations/" + padded_image + ".poses.pt")

        indices = torch.zeros((num_frames_total - num_frames), 2)
        if idx == 54:
                print('here')

        for i in range(num_frames_total - num_frames):
                start = i
                end = min(i + num_frames, num_frames_total)
        
                padded = str(current).zfill(8)
                current = current + 1

                indices = torch.zeros(3)
                indices[0] = start
                indices[1] = end
                indices[2] = idx

                torch.save(indices, "/data/mjakobs/data/jhmdb_fragments/indices/" + padded + ".indices.pt")
                
        #print("{} / {}".format(idx, length))
