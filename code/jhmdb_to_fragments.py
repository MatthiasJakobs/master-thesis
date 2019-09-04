from datasets import *
from visualization import show_predictions_ontop, show_pose_on_image
import matplotlib.pyplot as plt

import skimage.io as io

ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=False, use_saved_tensors=True)

length = len(ds)
for idx, entry in enumerate(ds):
        frames = entry["normalized_frames"]
        poses = entry["normalized_poses"]
        actions = entry["action_1h"]
        sequence_length = entry["sequence_length"]

        num_frames = 16

        num_frames_total = sequence_length[0]
        mini_batches = int(num_frames_total / num_frames) + 1

        for i in range(num_frames_total - num_frames):
                start = i
                end = min(i + num_frames, num_frames_total)

                mini_frames = frames[start:end]
                mini_poses = poses[start:end]

                assert len(mini_frames == num_frames)
                assert len(mini_poses == num_frames)

                padded = str(idx).zfill(8)
                
                torch.save(mini_frames, "/data/mjakobs/data/jhmdb_fragments/images/" + padded + ".frames.pt")
                torch.save(mini_poses, "/data/mjakobs/data/jhmdb_fragments/annotations/" + padded + ".poses.pt")
                torch.save(actions, "/data/mjakobs/data/jhmdb_fragments/annotations/" + padded + ".action_1h.pt")

        print("{} / {}".format(idx, length))
