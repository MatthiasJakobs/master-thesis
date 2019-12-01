import glob
import csv
import os
import torch

from datasets.JHMDBFragmentsDataset import JHMDBFragmentsDataset
from deephar.models import DeepHar

weights_folder = "/data/mjakobs/data/e2e_weights/"
output_file = weights_folder + "validation_new.csv"

weight_files = sorted(glob.glob(weights_folder + "weights_*"))

if os.path.exists(output_file):
    os.remove(output_file)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

ds = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=True, use_random_parameters=False, use_gt_bb=True)

with open(output_file, mode="a+") as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["iteration", "accuracy", "pose_model_accuracy", "visual_model_accuracy"])
    csv_file.flush()

for pretrained in weight_files:
    model = DeepHar(num_actions=21, num_frames=16, nr_context=2, use_gt=False, use_timedistributed=True)
    model.load_state_dict(torch.load(pretrained, map_location=device))
    model.eval()

    action_train_accuracies = []
    pose_train_accuracies = []
    iteration = 1000

    with torch.no_grad():
        correct = 0
        total = 0
        pose_model_correct = 0
        visual_model_correct = 0

        for i in range(len(ds)):
            validation_objects = ds[i]
            frames = validation_objects["frames"].to(device)
            actions = validation_objects["action_1h"].to(device)
            
            gt_pose = None
            
            ground_class = torch.argmax(actions)
            frames = frames.unsqueeze(0)

            assert len(frames) == 1

            _, predicted_poses, pose_model_pred, visual_model_pred, prediction = model(frames, gt_pose=gt_pose)

            pose_model_correct = pose_model_correct + (torch.argmax(pose_model_pred[0][-1]) == ground_class).item()
            visual_model_correct = visual_model_correct + (torch.argmax(visual_model_pred[0][-1]) == ground_class).item()

            if torch.cuda.is_available():
                frames = frames.cpu()
                actions = actions.cpu()
                predicted_poses = predicted_poses.cpu()

            pred_class = torch.argmax(prediction.squeeze(1), 1)

            total = total + len(pred_class)
            correct = correct + torch.sum(pred_class == ground_class).item()

            del frames
            del actions
            del predicted_poses
            del prediction

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        accuracy = correct / float(total)

        pose_model_accuracy = pose_model_correct / float(total)
        visual_model_accuracy = visual_model_correct / float(total)

        with open(output_file, mode="a+") as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([iteration, accuracy, pose_model_accuracy, visual_model_accuracy])
            csv_file.flush()

        iteration += 1000

