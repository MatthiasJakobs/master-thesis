from datasets.JHMDBDataset import JHMDBDataset
from deephar.models import DeepHar
from deephar.measures import categorical_cross_entropy

import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data

import torch
import torch.optim as optim
import torch.nn as nn

ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=False, use_saved_tensors=False)

if torch.cuda.is_available():
    device = 'cuda'
    cuda = True
else:
    device = 'cpu'
    cuda = False

model = DeepHar(num_actions=21, use_gt=False).to(device)

limit_data_percent = 1
validation_amount = 0.01
batch_size = 1 #TODO: Fix issue where two clips in the same batch cannot have different number of frames
val_batch_size = batch_size

number_of_datapoints = int(len(ds) * limit_data_percent)
indices = list(range(number_of_datapoints))
split = int((1 - validation_amount) * number_of_datapoints)

np.random.seed(3004)
np.random.shuffle(indices)

train_indices = indices[:split]
val_indices = indices[split:]

print("Using {} training and {} validation datapoints".format(len(train_indices), len(val_indices)))

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = data.DataLoader(
    ds,
    batch_size=batch_size,
    sampler=train_sampler
)

val_loader = data.DataLoader(
    ds,
    batch_size=val_batch_size,
    sampler=val_sampler
)

optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.98, nesterov=True)

iteration = 0
nr_epochs = 50

for epoch in range(nr_epochs):

    model.train()
    for batch_idx, train_objects in enumerate(train_loader):

        frames = train_objects["normalized_frames"].to(device)
        poses = train_objects["normalized_poses"].to(device)
        actions = train_objects["action_1h"].to(device)
        sequence_length = train_objects["sequence_length"].to(device)

        actions = actions.unsqueeze(1)
        actions = actions.expand(-1, 4, -1)

        num_frames = 16
        num_joints = 16

        num_frames_total = sequence_length[0]
        mini_batches = int(num_frames_total / num_frames) + 1
        losses = 0

        for i in range(num_frames_total - num_frames):
            start = i
            end = min(i + num_frames, num_frames_total)

            mini_frames = frames[:, start:end]
            mini_poses = poses[:, start:end, :, 0:2]

            mini_poses = mini_poses.reshape(batch_size, 2, num_frames, num_joints)

            pose, pose_predicted_actions, vis_predicted_actions, final_prediction = model(mini_frames, mini_poses)

            partial_loss_pose = torch.sum(categorical_cross_entropy(pose_predicted_actions, actions))
            partial_loss_action = torch.sum(categorical_cross_entropy(vis_predicted_actions, actions))
            losses = partial_loss_pose + partial_loss_action

            losses.backward(retain_graph=False)

            optimizer.step()
            optimizer.zero_grad()

            iteration = iteration + 1

            print("batch {} iteration {} loss {}".format(batch_idx, iteration, losses))

        if batch_idx % 1 == 0:
            # evaluate

            model.eval()
            with torch.no_grad():
                accuracy = 0

                for batch_idx, validation_objects in enumerate(val_loader):
                    frames = validation_objects["normalized_frames"].to(device)
                    sequence_length = validation_objects["sequence_length"].to(device)
                    actions = validation_objects["action_1h"].to(device)

                    num_frames = 16
                    num_frames_total = sequence_length[0]

                    ground_class = torch.argmax(actions[0])

                    num_miniframes = 0
                    for i in range(num_frames_total - num_frames):
                        start = i
                        end = min(i + num_frames, num_frames_total)

                        mini_frames = frames[:, start:end]
                        _, _, _, prediction = model(mini_frames, None)

                        pred_class = torch.argmax(prediction)

                        num_miniframes = num_miniframes + 1
                        accuracy = accuracy + int(pred_class == ground_class)
                        print("done with one 16 frame batch")

                print("accuracy", accuracy / float(num_miniframes))










