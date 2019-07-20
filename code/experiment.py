import torch.utils.data as data
import numpy as np

from timing import Timer

import csv
from datetime import datetime

from os import makedirs, remove
from os.path import exists

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from deephar.models import *
from deephar.utils import get_valid_joints
from deephar.measures import elastic_net_loss_paper
from deephar.evaluation import eval_pckh_batch
from datasets import MPIIDataset
from visualization import show_predictions_ontop

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_if_not_exists(path):
    if not exists(path):
        makedirs(path)

def remove_if_exists(path):
    if exists(path):
        remove(path)

def val_collate_fn(data):
    images = []
    poses = []
    headsizes = []
    matrices = []
    original_sizes = []
    original_images = []

    for obj in data:
        normalized_image = obj["normalized_image"].reshape(3, 256, 256)
        normalized_pose = obj["normalized_pose"]

        headsize = torch.from_numpy(obj["head_size"]).float().to(device)
        trans_matrix = torch.from_numpy(obj["trans_matrix"]).float().to(device)
        image_tensor = torch.from_numpy(normalized_image).float().to(device)
        pose_tensor = torch.from_numpy(normalized_pose).float().to(device)
        original_size = obj["original_size"].int().to(device)
        original_image = obj["original_image"]

        original_images.append(original_image)

        images.append(image_tensor)
        poses.append(pose_tensor)
        headsizes.append(headsize)
        matrices.append(trans_matrix)
        original_sizes.append(original_size)

    t_images = torch.stack(images, dim=0)
    t_poses = torch.stack(poses, dim=0)
    t_headsizes = torch.stack(headsizes, dim=0)
    t_matrices = torch.stack(matrices, dim=0)
    t_sizes = torch.stack(original_sizes, dim=0)
    return t_images, t_poses, t_headsizes, t_matrices, t_sizes, original_images

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

def run_experiment_mpii(conf):
    learning_rate = conf["learning_rate"]
    nr_epochs = conf["nr_epochs"]
    validation_amount = conf["validation_amount"]
    limit_data_percent = conf["limit_data_percent"]
    numpy_seed = conf["numpy_seed"]
    num_blocks = conf["num_blocks"]
    name = conf["name"]
    batch_size = conf["batch_size"]
    val_batch_size = conf["val_batch_size"]

    ds = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=False)

    if num_blocks == 1:
        model = Mpii_1().to(device)
    elif num_blocks == 2:
        model = Mpii_2().to(device)
    elif num_blocks == 4:
        model = Mpii_4().to(device)
    elif num_blocks == 8:
        model = Mpii_8().to(device)

    number_of_datapoints = int(len(ds) * limit_data_percent)
    indices = list(range(number_of_datapoints))
    split = int((1 - validation_amount) * number_of_datapoints)

    np.random.seed(numpy_seed)
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
        batch_size=val_batch_size,
        sampler=val_sampler,
        collate_fn=val_collate_fn
    )

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, verbose=True)

    if name is not None:
        experiment_name = name
    else:
        experiment_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    create_if_not_exists("experiments")
    create_if_not_exists("experiments/{}".format(experiment_name))
    create_if_not_exists("experiments/{}/weights".format(experiment_name))

    remove_if_exists("experiments/{}/validation.csv".format(experiment_name))

    with open('experiments/{}/parameters.csv'.format(experiment_name), 'w+') as parameter_file:
        parameter_file.write("paramter_name,value\n")
        parameter_file.write("learning_rate,{}\n".format(learning_rate))
        parameter_file.write("batch_size,{}\n".format(batch_size))
        parameter_file.write("number_of_datapoints,{}\n".format(number_of_datapoints))
        parameter_file.write("limit_data_percent,{}\n".format(limit_data_percent))
        parameter_file.write("numpy_seed,{}\n".format(numpy_seed))
        parameter_file.write("num_blocks,{}\n".format(num_blocks))
        parameter_file.write("nr_epochs,{}\n".format(nr_epochs))

    t_train_epoch = Timer("train_epoch", experiment_name)
    t_train_batch = Timer("train_batch", experiment_name)
    t_val_batch = Timer("val_batch", experiment_name)
    t_val_epoch = Timer("val_epoch", experiment_name)
    t_model = Timer("model", experiment_name)

    with open("experiments/{}/validation.csv".format(experiment_name), mode="w") as val_file:
        val_writer = csv.writer(val_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        val_writer.writerow(['iteration', 'pckh_0.5', 'pckh_0.2'])


    with open('experiments/{}/loss.csv'.format(experiment_name), mode='w') as output_file:
        writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['epoch', 'batch_nr', 'iteration', 'loss'])


        iteration = 0
        for epoch in range(nr_epochs):
            t_train_epoch.start()

            model.train()
            for batch_idx, (images, poses) in enumerate(train_loader):
                t_train_batch.start()

                images = images
                poses = poses

                t_model.start()
                _, output = model(images)
                t_model.stop()

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

                pose_loss = elastic_net_loss_paper(pred_pose, ground_pose)
                loss = vis_loss * 0.01 + pose_loss

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                t_train_batch.stop()
                print("epoch {} batch_nr {} loss {}".format(epoch, batch_idx, loss.item()))
                writer.writerow([epoch, batch_idx, iteration, loss.item()])
                output_file.flush()

                iteration = iteration + 1

            t_train_epoch.stop()

            val_accuracy_05 = []
            val_accuracy_02 = []

            scheduler.step(loss.item())

            model.eval()

            if not exists('experiments/{}/val_images'.format(experiment_name)):
                makedirs('experiments/{}/val_images'.format(experiment_name))

            with torch.no_grad():
                t_val_epoch.start()

                for batch_idx, (val_images, val_poses, val_headsizes, val_trans_matrices, val_original_sizes, val_orig_images) in enumerate(val_loader):
                    t_val_batch.start()

                    heatmaps, predictions = model(val_images)
                    predictions = predictions[-1, :, :, :].squeeze(dim=0)

                    if predictions.dim() == 2:
                        predictions = predictions.unsqueeze(0)

                    if not exists('experiments/{}/val_images/{}'.format(experiment_name, epoch)):
                        makedirs('experiments/{}/val_images/{}'.format(experiment_name, epoch))

                    show_predictions_ontop(val_poses[0], val_orig_images[0], predictions[0], 'experiments/{}/val_images/{}/{}.png'.format(experiment_name, epoch, batch_idx), val_trans_matrices[0], val_original_sizes[0])

                    scores_05, scores_02 = eval_pckh_batch(predictions, val_poses, val_headsizes, val_trans_matrices)
                    val_accuracy_05.extend(scores_05)
                    val_accuracy_02.extend(scores_02)

                    t_val_batch.stop()



                mean_05 = np.mean(np.array(val_accuracy_05))
                mean_02 = np.mean(np.array(val_accuracy_02))

                print([iteration, loss.item(), mean_05, mean_02])

                with open("experiments/{}/validation.csv".format(experiment_name), mode="a") as val_file:
                    val_writer = csv.writer(val_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    val_writer.writerow([iteration, mean_05, mean_02])

                t_val_epoch.stop()

                torch.save(model.state_dict(), "experiments/{}/weights/weights_{:04d}".format(experiment_name, epoch))

    t_train_epoch.save()
    t_train_batch.save()
    t_val_batch.save()
    t_val_epoch.save()
    t_model.save()
