from datasets.MPIIDataset import MPIIDataset

from skimage import io
from skimage.transform import resize

from deephar.image_processing import rotate_and_crop
from deephar.utils import scale, transform_pose, translate, flip_h, flip_lr_pose
from deephar.models import Mpii_8
import numpy as np
import torch

import os
import csv

import matplotlib.pyplot as plt

def preprocess(old_scale, center_x, center_y, full_image_path):
    new_scale = old_scale * 1.25 # magic value
    new_objpose = np.array([center_x, center_y + 12 * new_scale]) # magic values, no idea where they are comming from

    window_size = new_scale * 200

    bbox = torch.IntTensor([
        new_objpose[0] - window_size / 2, # x1, upper left
        new_objpose[1] - window_size / 2, # y1, upper left
        new_objpose[0] + window_size / 2, # x2, lower right
        new_objpose[1] + window_size / 2  # y2, lower right
    ])

    center = torch.from_numpy(new_objpose).float()
    window_size = [window_size, window_size]

    image = torch.IntTensor(io.imread(full_image_path))
    plt.imshow(image.byte())

    trans_matrix, image = rotate_and_crop(image, torch.FloatTensor([0.0]), center, window_size)
    size_after_rotate = torch.FloatTensor([image.shape[1], image.shape[0]])
    image = torch.FloatTensor(resize(image, (255, 255), preserve_range=True))
    image = 2.0 * (image / 255.0) - 1.0 
    trans_matrix = scale(trans_matrix, 255 / size_after_rotate[0], 255 / size_after_rotate[1])
    trans_matrix = scale(trans_matrix, 1.0 / 255, 1.0 / 255)

    image = image.permute(2, 0, 1)

    return image, trans_matrix

def main():
    ds = MPIIDataset("/data/mjakobs/data/mpii/", train=False, use_random_parameters=False, use_saved_tensors=False)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = Mpii_8(num_context=2).to(device)
    model.load_state_dict(torch.load("mpii_8_trained", map_location=device))

    if os.path.exists("mpii_test.csv"):
        os.remove("mpii_test.csv")
    
    with open("mpii_test.csv", mode="a+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "imgidx",
                "ridx",
                "r_ankle_x",
                "r_ankle_y",
                "r_knee_x",
                "r_knee_y",
                "r_hip_x",
                "r_hip_y",
                "l_hip_x",
                "l_hip_y",
                "l_knee_x",
                "l_knee_y",
                "l_ankle_x",
                "l_ankle_y",
                "pelvis_x",
                "pelvis_y",
                "thorax_x",
                "thorax_y",
                "upper_neck_x",
                "upper_neck_y",
                "head_top_x",
                "head_top_y",
                "r_wrist_x",
                "r_wrist_y",
                "r_elbow_x",
                "r_elbow_y",
                "r_shoulder_x",
                "r_shoulder_y",
                "l_shoulder_x",
                "l_shoulder_y",
                "l_elbow_x",
                "l_elbow_y",
                "l_wrist_x",
                "l_wrist_y"
            ]
        )
        csv_file.flush()

    for idx in range(len(ds)):
        entry = ds[idx]
        print(idx)
        if entry["rects"] == [[]] or [] in entry["rects"]:
            continue
        image_name = entry["image"]
        rects = entry["rects"]
        image_id = entry["imgidx"]
        rect_ids = entry["ridx"]

        full_image_path = "/data/mjakobs/data/mpii/images/" + image_name

        for rid, rect in enumerate(rects):
            old_scale = rect[0]
            center_x = rect[1]
            center_y = rect[2]

            image, trans_matrix = preprocess(old_scale, center_x, center_y, full_image_path)

            trans_matrix = torch.FloatTensor(trans_matrix)
            model_input = image.unsqueeze(0).to(device)
            _, predictions, _, _ = model(model_input)
            predictions = predictions.squeeze()
            
            transformed_pose = torch.FloatTensor(transform_pose(trans_matrix, predictions[:, 0:2], inverse=True))
            #plt.scatter(x=transformed_pose[:, 0], y=transformed_pose[:, 1])
            #plt.show()
            del predictions
            del model_input
            
            # FLIP LR
            image_flipped = torch.FloatTensor(np.fliplr(image.clone().permute(1, 2, 0))+1) - 1.0
            image_flipped = image_flipped.permute(2, 0, 1)
            model_input = image_flipped.unsqueeze(0).to(device)
            _, predictions, _, _ = model(model_input)
            predictions = predictions.squeeze()

            transformed_pose_flipped = predictions[:, 0:2]
            del predictions
            del model_input

            transformed_pose_flipped[:, 0] = 1.0 - transformed_pose_flipped[:, 0]  
            transformed_pose_flipped = torch.FloatTensor(transform_pose(trans_matrix, transformed_pose_flipped, inverse=True))
            transformed_pose_flipped = flip_lr_pose(transformed_pose_flipped)

            # # displace center positive
            # percentage = 10.0
            # new_center_x = center_x + (center_x / percentage)
            # new_center_y = center_y + (center_y / percentage)
            # image, trans_matrix = preprocess(old_scale, center_x, center_y, full_image_path)
            # model_input = image.unsqueeze(0)
            # _, predictions, _, _ = model(model_input)
            # predictions = predictions.squeeze()
            # positive_displacement = torch.FloatTensor(transform_pose(trans_matrix, predictions[:, 0:2], inverse=True))

            # # displace center negative
            # percentage = 10.0
            # new_center_x = center_x - (center_x / percentage)
            # print(center_x / percentage)
            # new_center_y = center_y - (center_y / percentage)
            # image, trans_matrix = preprocess(old_scale, center_x, center_y, full_image_path)
            # model_input = image.unsqueeze(0)
            # _, predictions, _, _ = model(model_input)
            # predictions = predictions.squeeze()
            # negative_displacement = torch.FloatTensor(transform_pose(trans_matrix, predictions[:, 0:2], inverse=True))

            # Calculate mean pose
            mean_pose = transformed_pose.clone()
            del transformed_pose
            mean_pose = mean_pose + transformed_pose_flipped.clone()
            del transformed_pose_flipped
            # mean_pose = mean_pose + positive_displacement.clone()
            # mean_pose = mean_pose + negative_displacement.clone()
            mean_pose = mean_pose / 2.0
            mean_pose = mean_pose.int()
            with open("mpii_test.csv", mode="a+") as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    [
                        idx + 1,
                        rect_ids[rid],
                        mean_pose[0, 0].item(),
                        mean_pose[0, 1].item(),
                        mean_pose[1, 0].item(),
                        mean_pose[1, 1].item(),
                        mean_pose[2, 0].item(),
                        mean_pose[2, 1].item(),
                        mean_pose[3, 0].item(),
                        mean_pose[3, 1].item(),
                        mean_pose[4, 0].item(),
                        mean_pose[4, 1].item(),
                        mean_pose[5, 0].item(),
                        mean_pose[5, 1].item(),
                        mean_pose[6, 0].item(),
                        mean_pose[6, 1].item(),
                        mean_pose[7, 0].item(),
                        mean_pose[7, 1].item(),
                        mean_pose[8, 0].item(),
                        mean_pose[8, 1].item(),
                        mean_pose[9, 0].item(),
                        mean_pose[9, 1].item(),
                        mean_pose[10, 0].item(),
                        mean_pose[10, 1].item(),
                        mean_pose[11, 0].item(),
                        mean_pose[11, 1].item(),
                        mean_pose[12, 0].item(),
                        mean_pose[12, 1].item(),
                        mean_pose[13, 0].item(),
                        mean_pose[13, 1].item(),
                        mean_pose[14, 0].item(),
                        mean_pose[14, 1].item(),
                        mean_pose[15, 0].item(),
                        mean_pose[15, 1].item(),
                    ]
                )
                csv_file.flush()
            del mean_pose

main()