import torch
import numpy as np
import matplotlib.pyplot as plt 

from datasets.JHMDBDataset import JHMDBDataset
from datasets.JHMDBDataset import actions as jhmdb_actions
from deephar.models import DeepHar

if torch.cuda.is_available:
    device = "cuda"
else:
    device = "cpu"

def check_e2e(model_path, confusion_classes):
    model = DeepHar(num_actions=21, use_gt=False, nr_context=2, use_timedistributed=True).to(device)

    with torch.no_grad():
        test_ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=False, use_gt_bb=True)
            model.load_state_dict(torch.load(model_path, map_location=device))

        self.model.eval()

        nr_confusions = 0

        for i in range(len(test_ds)):
            print(i)
            test_objects = test_ds[i]
            frames = test_objects["normalized_frames"].to(self.device)
            actions = test_objects["action_1h"].to(self.device)
            ground_poses = test_objects["normalized_poses"].to(self.device)
            sequence_length = test_objects["sequence_length"].to(self.device).item()
            padding = int((sequence_length - 16) / 2.0)
            if sequence_length < 16:
                print("length smaller than 16")
                continue

            ground_class = torch.argmax(actions, 0)
            single_clip = frames[padding:(16 + padding)]

            bboxes = test_objects["bbox"][padding:(16 + padding)]
            trans_matrices = test_objects["trans_matrices"][padding:(16 + padding)]
            ground_poses = ground_poses[padding:(16 + padding)]

            assert len(single_clip) == 16

            single_clip = single_clip.unsqueeze(0).to(self.device)
            _, predicted_poses, _, _, single_result = self.model(single_clip)

            pred_class_single = torch.argmax(single_result.squeeze(1))
            predicted_poses = predicted_poses.squeeze(0)

            if ground_class.item() == confusion_classes[1] and pred_class_single.item() == confusion_classes[0]:
                print("confusion detected")
                images = single_clip[::4]
                assert len(images) == 4
                poses = predicted_poses[::4]
                
                for i in range(4):
                    plt.subplot(2,2,i+1)
                    plt.imshow((images[i].permute(1,2,0) + 1 ) / 2.0)
                    plt.scatter(x=predicted_poses[i, :, 0] * 255.0 * predicted_poses[i, :, 2], y=predicted_poses[i, :, 1] * 255.0 * predicted_poses[i, :, 2], s=10, c="#00FFFF")

                plt.tight_layout()
                plt.savefig("confusion_e2e/said_{}_was_{}_{}.png".format(jhmdb_actions[confusion_classes[0]], jhmdb_actions[confusion_classes[1]], nr_confusions))
                nr_confusions += 1

def main():
    confusion = [16, 15] # model says stand but is actually sit
    check_e2e("/data/mjakobs/data/e2e_final_228000", confusion)

main()