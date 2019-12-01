import torch
import numpy as np
from datasets.JHMDBDataset import JHMDBDataset
from deephar.models import Mpii_4
from deephar.evaluation import eval_pck_batch

pretrained_model = "/data/mjakobs/data/pose_jhmdb_perjoint"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

with torch.no_grad():
    test_ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=False, use_gt_bb=True)
    model = Mpii_4(num_context=0).to(device)
    model.load_state_dict(torch.load(pretrained_model, map_location=device))

    model.eval()

    per_joint_accuracy = np.zeros(16)
    number_valids = np.zeros(16)

    for i in range(len(test_ds)):
        print(i)
        test_objects = test_ds[i]
        frames = test_objects["normalized_frames"].to(device)
        ground_poses = test_objects["normalized_poses"].to(device)
        sequence_length = test_objects["sequence_length"].to(device).item()
        padding = int((sequence_length - 16) / 2.0)
        if sequence_length < 16:
            print("length smaller than 16")
            continue

        single_clip = frames[padding:(16 + padding)]

        bboxes = test_objects["bbox"][padding:(16 + padding)]
        trans_matrices = test_objects["trans_matrices"][padding:(16 + padding)]
        ground_poses = ground_poses[padding:(16 + padding)]

        assert len(single_clip) == 16

        single_clip = single_clip.to(device)
        _, predicted_poses, _, _= model(single_clip)

        predicted_poses = predicted_poses.squeeze(0)

        distance_meassures = torch.FloatTensor(len(bboxes))

        for i in range(len(bboxes)):
            width = torch.abs(bboxes[i, 0] - bboxes[i, 2])
            height = torch.abs(bboxes[i, 1] - bboxes[i, 3])

            distance_meassures[i] = torch.max(width, height).item()

        matches, valids = eval_pck_batch(predicted_poses[:, :, 0:2], ground_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.1, return_perjoint=True)

        number_valids = number_valids + np.array(valids[0])
        for u in range(16):
            if valids[i][u]:
                per_joint_accuracy[u] = per_joint_accuracy[u] + matches[0][u]

    number_valids[7] = 1 # joint never visible
    per_joint_final = per_joint_accuracy / number_valids
    print(per_joint_final)
    left_arm = per_joint_final[13] + per_joint_final[14] + per_joint_final[15]
    right_arm = per_joint_final[10] + per_joint_final[11] + per_joint_final[12]
    right_leg = per_joint_final[0] + per_joint_final[1] + per_joint_final[2]
    left_leg = per_joint_final[3] + per_joint_final[4] + per_joint_final[5]
    legs_both = left_leg + right_leg
    arms_both = left_arm + right_arm
    upper_body_total = arms_both + per_joint_final[8] + per_joint_final[9]
    lower_body_total = legs_both + per_joint_final[6]
    print("left_arm", str(left_arm / 3))
    print("right_arm", str(right_arm / 3))
    print("left_leg", str(left_leg / 3))
    print("right_leg", str(right_leg / 3))
    print("legs_both", str(legs_both / 6))
    print("arms_both", str(arms_both / 6))
    print("upper_body", str(upper_body_total / 8))
    print("lower_body", str(lower_body_total / 7))
