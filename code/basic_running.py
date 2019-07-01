import torch.utils.data as data
import numpy as np

import csv
from datetime import datetime

from os import makedirs
from os.path import exists

import torch
import torch.optim as optim
import torch.nn as nn

from deephar.models import Mpii_No_Context
from deephar.utils import get_valid_joints
from datasets import MPIIDataset

from socket import gethostname

ds = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Mpii_No_Context().to(device)

def collate_fn(data):
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
    #print("t_poses size", t_poses.size())
    #print("t_images size", t_images.size())
    return t_images, t_poses

if gethostname() == "ares":
    batch_size = 70
else:
    batch_size = 10

learning_rate = 0.00001

train_loader = data.DataLoader(
    ds,
    num_workers=0,
    batch_size=batch_size,
    collate_fn=collate_fn
)

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

def elastic_net_loss(y_pred, y_true):
    # note: not the one used in code but the one in the paper
    valid, nr_valid = get_valid_joints(y_true)

    y_pred = y_pred * valid
    y_true = y_true * valid

    l1 = torch.sum(torch.abs(y_pred - y_true), (1, 2))
    l2 = torch.sum(torch.pow(y_pred - y_true, 2), (1, 2))

    final_losses = (l1 + l2) / nr_valid
    loss = torch.sum(final_losses)
    return loss

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

if not exists("experiments"):
    makedirs("experiments")

makedirs("experiments/{}".format(timestamp))

with open('experiments/{}/parameters.csv'.format(timestamp), 'w+') as parameter_file:
    parameter_file.write("learning_rate={}\n".format(learning_rate))
    parameter_file.write("batch_size={}\n".format(batch_size))


with open('experiments/{}/loss.csv'.format(timestamp), mode='w') as output_file:
    writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['epoch', 'batch_nr', 'loss'])
    model.train()

    for epoch in range(10):
        for batch_idx, (images, poses) in enumerate(train_loader):
            images = images
            poses = poses
            
            output = model(images)
            # output shape: (batch_size, 16, 3)
            pred_pose = output[:, :, 0:2]
            ground_pose = poses[:, :, 0:2]

            pred_vis = output[:, :, 2]
            ground_vis = poses[:, :, 2]

            binary_crossentropy = nn.BCELoss()

            vis_loss = binary_crossentropy(pred_vis, ground_vis)
            #vis_loss.backward(retain_graph=True)

            pose_loss = elastic_net_loss(pred_pose, ground_pose)
            loss = vis_loss * 0.01 + pose_loss         
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                writer.writerow([epoch, batch_idx, loss.item()])
                output_file.flush()
                print("epoch {} batch_nr {} loss {}".format(epoch, batch_idx, loss.item()))

    torch.save(model.state_dict(), "experiments/{}/weights_{}".format(timestamp, epoch))

