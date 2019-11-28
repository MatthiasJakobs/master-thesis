import torch
import matplotlib.pyplot as plt 

from skimage import io

from deephar.models import DeepHar, Mpii_8
from deephar.utils import transform_pose
from deephar.layers import TimeDistributedPoseEstimation
from datasets.PennActionDataset import PennActionDataset
from datasets.JHMDBDataset import JHMDBDataset
from datasets.MPIIDataset import MPIIDataset

joint_mapping = [
    [0, 1],
    [1, 2],
    [2, 6],
    [3, 6],
    [4, 3],
    [5, 4],
    [6, 7],
    [7, 8],
    [8, 9],
    [10, 11],
    [11, 12],
    [12, 7],
    [13, 7],
    [14, 13],
    [15, 14]
]

def plot(pose, matrix, path):

    # plt.tick_params(
    # axis='both',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # right=False,
    # left=False,
    # top=False,         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off
    plt.axis("off")

    vis = pose[:, 2] > 0.5
    pose = pose[:, 0:2]
    pose = torch.from_numpy(transform_pose(matrix, pose, inverse=True)).float()

    pred_x = pose[:, 0]
    pred_y = pose[:, 1]

    s = 10

    for _, (src, dst) in enumerate(joint_mapping):
        if not vis[src]:
            continue

        if vis[dst]:
            plt.plot([pred_x[src], pred_x[dst]], [pred_y[src], pred_y[dst]], lw=1, c="#00FF00")


    plt.scatter(x=pred_x[0:3], y=pred_y[0:3], c="#FF00FF", s=s) # right
    plt.scatter(x=pred_x[10:13], y=pred_y[10:13], c="#FF00FF", s=s) # right
    
    plt.scatter(x=pred_x[3:6], y=pred_y[3:6], c="#FFFF00", s=s) # left
    plt.scatter(x=pred_x[13:16], y=pred_y[13:16], c="#FFFF00", s=s) # left

    plt.scatter(x=pred_x[6:10], y=pred_y[6:10], c="#00FFFF", s=s) # rest
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def extract_har_jhmdb_pose(model_path, output_path):
    ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=False)
    model = DeepHar(num_actions=21, use_gt=True, nr_context=2, model_path="/data/mjakobs/data/pretrained_mixed_pose", use_timedistributed=True).to("cpu")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    for i in range(len(ds)):
        entry = ds[i]
        sequence_length = entry["sequence_length"][0].item()
        frames = entry["normalized_frames"][:sequence_length]
        matrices = entry["trans_matrices"][:sequence_length]
        all_frames = entry["all_frames"]
        frames = frames.unsqueeze(0)
        _, predicted_poses, _, _, _ = model(frames, gt_pose=None)

        for frame in range(sequence_length):
            path = '{}/{}_{}.png'.format(output_path, i, frame)
            image = io.imread(all_frames[frame])
            plt.imshow(image)

            pose = predicted_poses.squeeze(0)
            pose = pose[frame]

            plot(pose, matrices[frame], path)

def extract_har_pennaction_pose(model_path, output_path):
    ds = PennActionDataset("/data/mjakobs/data/pennaction/", train=False)
    model = DeepHar(num_actions=15, use_gt=False, nr_context=2, use_timedistributed=True).to("cpu")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    for i in [10]:
        entry = ds[i]
        sequence_length = entry["sequence_length"][0].item()
        frames = entry["normalized_frames"]
        matrices = entry["trans_matrices"]
        frame_folder = entry["frame_folder"]
        frames = frames.unsqueeze(0)
        _, predicted_poses, _, _, _ = model(frames, gt_pose=None)

        for frame in range(sequence_length):
            frame_index = "{}".format(frame + 1).zfill(6)
            path = '{}/{}_{}.png'.format(output_path, i, frame)
            image = io.imread(frame_folder + frame_index + ".jpg")
            plt.imshow(image)

            pose = predicted_poses.squeeze(0)
            pose = pose[frame]

            plot(pose, matrices[frame], path)


def extract_jhmdb_pose(model_path, output_path):
    ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=False)
    model = Mpii_8(num_context=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    for i in [100]:
        entry = ds[i]
        sequence_length = entry["sequence_length"][0].item()
        frames = entry["normalized_frames"][:sequence_length]
        matrices = entry["trans_matrices"][:sequence_length]
        all_frames = entry["all_frames"]
        _, predicted_poses, _, _ = model(frames)

        for frame in range(sequence_length):
            path = '{}/{}_{}.png'.format(output_path, i, frame)
            image = io.imread(all_frames[frame])
            plt.imshow(image)

            pose = predicted_poses.squeeze(0)
            pose = pose[frame]

            plot(pose, matrices[frame], path)

def extract_mpii_pose(model_path, output_path):
    ds = MPIIDataset("/data/mjakobs/data/mpii/", train=True, val=False, use_random_parameters=False)
    model = Mpii_8(num_context=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    for i in range(len(ds)):
        entry = ds[i]
        frames = entry["normalized_image"].unsqueeze(0)
        matrices = entry["trans_matrix"]
        image_number = entry["image_path"].item()
        image_number = "{}".format(image_number).zfill(9)
        image_path = "/data/mjakobs/data/mpii/images/{}.jpg".format(image_number)
        _, predicted_poses, _, _ = model(frames)

        path = '{}/{}.png'.format(output_path, i)
        image = io.imread(image_path)
        plt.imshow(image)

        pose = predicted_poses.squeeze(0).squeeze(0)

        plot(pose.detach(), matrices, path)


def main():
    #extract_jhmdb_pose("/tmp/jhmdb_8_model", "/tmp/jhmdb_images")
    extract_mpii_pose("/tmp/mpii_8_model", "/tmp/mpii_images")
    #extract_har_jhmdb_pose("/tmp/har_jhmdb_nof", "/tmp/har_jhmdb_images")
    #extract_har_pennaction_pose("/tmp/har_penn_model", "/tmp/har_penn_images")

main()