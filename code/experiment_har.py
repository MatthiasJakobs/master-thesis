from datasets import JHMDBDataset
from deephar.models import DeepHar
from deephar.measures import categorical_cross_entropy

import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data

import torch
import torch.optim as optim
import torch.nn as nn

from torchviz import make_dot

ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", use_random_parameters=False, use_saved_tensors=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeepHar(num_actions=21, use_gt=False).to(device)

limit_data_percent = 0.01
validation_amount = 0.1
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

for epoch in range(3):

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

            pose, predicted_actions = model(mini_frames, mini_poses)

            partial_loss = torch.sum(categorical_cross_entropy(predicted_actions, actions))
            losses = losses + partial_loss

        losses.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        iteration = iteration + 1

        print("iteration {} loss {}".format(iteration, losses))

        # evaluate
