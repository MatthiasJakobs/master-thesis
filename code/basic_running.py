import torch.utils.data as data
import numpy as np

import csv
from datetime import datetime

from os import makedirs
from os.path import exists

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from deephar.models import *
from deephar.utils import get_valid_joints
from deephar.evaluation import eval_pckh_batch
from datasets import MPIIDataset

from socket import gethostname

ds = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=False)

def val_collate_fn(data):
    images = []
    poses = []
    headsizes = []
    matrices = []

    for obj in data:
        normalized_image = obj["normalized_image"].reshape(3, 256, 256)
        normalized_pose = obj["normalized_pose"]

        headsize = torch.from_numpy(obj["head_size"]).float().to(device)
        trans_matrix = torch.from_numpy(obj["trans_matrix"]).float().to(device)
        image_tensor = torch.from_numpy(normalized_image).float().to(device)
        pose_tensor = torch.from_numpy(normalized_pose).float().to(device)

        images.append(image_tensor)
        poses.append(pose_tensor)
        headsizes.append(headsize)
        matrices.append(trans_matrix)

    t_images = torch.stack(images, dim=0)
    t_poses = torch.stack(poses, dim=0)
    t_headsizes = torch.stack(headsizes, dim=0)
    t_matrices = torch.stack(matrices, dim=0)
    return t_images, t_poses, t_headsizes, t_matrices

def train_collate_fn(data):
    # data = [output_obj1, ..., output_obj_n]
    images = []
    poses = []
    for obj in data:
        normalized_image = obj["normalized_image"].reshape(3, 256, 256)
        normalized_pose = obj["normalized_pose"]

        image_tensor = torch.from_numpy(normalized_image).float().to(device)
        pose_tensor = torch.from_numpy(normalized_pose).float().to(device)
        images.append(image_tensor)
        poses.append(pose_tensor)

    t_images = torch.stack(images, dim=0)
    t_poses = torch.stack(poses, dim=0)

    return t_images, t_poses

if gethostname() == "ares" or gethostname() == "prometheus":
    batch_size = 40
else:
    batch_size = 2

learning_rate = 0.00001
nr_epochs = 10
validation_amount = 0.1 # 10 percent
limit_data_percent = 1 # limit dataset to x percent (for testing)
random_seed = 30004
num_blocks = 1
name = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Mpii_1().to(device)

number_of_datapoints = int(len(ds) * limit_data_percent)
indices = list(range(number_of_datapoints))
split = int((1 - validation_amount) * number_of_datapoints)

np.random.seed(random_seed)
np.random.shuffle(indices)

train_indices = indices[:split]
val_indices = indices[split:]

print("Using {} training and {} validation datapoints".format(len(train_indices), len(val_indices)))

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = data.DataLoader(
    ds,
    num_workers=0,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=train_collate_fn
)

val_loader = data.DataLoader(
    ds,
    num_workers=0,
    batch_size=batch_size,
    sampler=val_sampler,
    collate_fn=val_collate_fn
)

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

def elastic_net_loss_paper(y_pred, y_true):
    # note: not the one used in code but the one in the paper

    valid, nr_valid = get_valid_joints(y_true[:, 0, :, :])
    valid = valid.unsqueeze(1)
    valid = valid.expand(-1, y_pred.size()[1], -1, -1)
    y_pred = y_pred * valid
    y_true = y_true * valid

    l1 = torch.sum(torch.abs(y_pred - y_true), (1, 2, 3))
    l2 = torch.sum(torch.pow(y_pred - y_true, 2), (1, 2, 3))

    final_losses = (l1 + l2) / nr_valid
    loss = torch.sum(final_losses)
    return loss

if name is not None:
    experiment_name = name
else:
    experiment_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

if not exists("experiments"):
    makedirs("experiments")

if not exists("experiments/{}".format(experiment_name)):
    makedirs("experiments/{}".format(experiment_name))

if not exists("experiments/{}/weights".format(experiment_name)):
    makedirs("experiments/{}/weights".format(experiment_name))

with open('experiments/{}/parameters.csv'.format(experiment_name), 'w+') as parameter_file:
    parameter_file.write("paramter_name,value\n")
    parameter_file.write("learning_rate,{}\n".format(learning_rate))
    parameter_file.write("batch_size,{}\n".format(batch_size))
    parameter_file.write("number_of_datapoints,{}\n".format(number_of_datapoints))
    parameter_file.write("limit_data_percent,{}\n".format(limit_data_percent))
    parameter_file.write("numpy_seed,{}\n".format(random_seed))
    parameter_file.write("num_blocks,{}\n".format(num_blocks))
    parameter_file.write("nr_epochs,{}\n".format(nr_epochs))


with open('experiments/{}/loss.csv'.format(experiment_name), mode='w') as output_file:
    writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['epoch', 'batch_nr', 'loss', 'pckh_0.5', 'pckh_0.2'])

    for epoch in range(nr_epochs):
        for batch_idx, (images, poses) in enumerate(train_loader):
            model.train()
            images = images
            poses = poses
            
            _, output = model(images)
            output = output.view(images.size()[0], num_blocks, -1, 3)
            # output shape: (batch_size, num_blocks, 16, 3)
            pred_pose = output[:, :, :, 0:2]
            ground_pose = poses[:, :, 0:2]
            ground_pose = ground_pose.unsqueeze(1)
            ground_pose = ground_pose.expand(-1, num_blocks, -1, -1)


            pred_vis = output[:, :, :, 2]
            ground_vis = poses[:, :, 2]
            ground_vis = ground_vis.unsqueeze(1)
            ground_vis = ground_vis.expand(-1, num_blocks, -1)

            binary_crossentropy = nn.BCELoss()

            vis_loss = binary_crossentropy(pred_vis, ground_vis)
            #vis_loss.backward(retain_graph=True)

            pose_loss = elastic_net_loss_paper(pred_pose, ground_pose)
            loss = vis_loss * 0.01 + pose_loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            print("epoch {} batch_nr {} loss {}".format(epoch, batch_idx, loss.item()))

        val_accuracy_05 = []
        val_accuracy_02 = []

        for batch_idx, (val_images, val_poses, val_headsizes, val_trans_matrices) in enumerate(val_loader):
            scores_05, scores_02 = eval_pckh_batch(model, val_images, val_poses, val_headsizes, val_trans_matrices)
            val_accuracy_05.extend(scores_05)
            val_accuracy_02.extend(scores_02)

        writer.writerow([epoch, batch_idx, loss.item(), np.mean(np.array(val_accuracy_05)), np.mean(np.array(val_accuracy_02))])
        print([epoch, batch_idx, loss.item(), np.mean(np.array(val_accuracy_05)), np.mean(np.array(val_accuracy_02))])
        output_file.flush()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "experiments/{}/weights/weights_{:04d}".format(experiment_name, epoch))

    torch.save(model.state_dict(), "experiments/{}/weights/weights_final".format(experiment_name))

