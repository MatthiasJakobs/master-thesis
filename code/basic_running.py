import torch.utils.data as data
import numpy as np

import csv
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn

from deephar.models import Mpii_No_Context
from datasets import MPIIDataset

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

train_loader = data.DataLoader(
   ds,
   num_workers=0,
   batch_size=30,
   pin_memory=False,
   collate_fn=collate_fn
)

optimizer = optim.RMSprop(model.parameters(), lr=0.001)

def elastic_net_loss(y_pred, y_true):
   # note: not the one used in code but the one in the paper
   l1 = torch.sum(torch.abs(y_pred - y_true))
   l2 = torch.sum(torch.pow(y_pred - y_true, 2))
   return (l1 + l2) / len(y_pred[1])

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

with open('logs/basic_running_{}.csv'.format(timestamp), mode='w') as output_file:

   writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
   writer.writerow(['epoch', 'batch_nr', 'loss'])
   for epoch in range(10):
      model.train()
      for batch_idx, (images, poses) in enumerate(train_loader):
         images = images
         poses = poses

         optimizer.zero_grad()
         
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

         if batch_idx % 10 == 0:
            writer.writerow([epoch, batch_idx, loss.item()])
            output_file.flush()
            print("epoch {} batch_nr {} loss pose {} loss vis {}".format(epoch, batch_idx, pose_loss.item(), vis_loss.item()))
      

