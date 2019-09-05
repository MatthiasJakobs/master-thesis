from datetime import datetime

from shutil import rmtree

from os import makedirs, remove
from os.path import exists

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

import csv
import numpy as np

from skimage import io

from datasets.MPIIDataset import *
from datasets.JHMDBFragmentsDataset import JHMDBFragmentsDataset
from deephar.models import DeepHar, Mpii_1, Mpii_2, Mpii_4, Mpii_8
from deephar.utils import get_valid_joints
from deephar.measures import elastic_net_loss_paper, categorical_cross_entropy
from deephar.evaluation import eval_pckh_batch

from visualization import show_predictions_ontop, visualize_heatmaps

class CSVWriter:
    def __init__(self, experiment_name, file_name):
        self.experiment_name = experiment_name
        self.file_name = file_name

    def write(self, row):
        with open("experiments/{}/{}.csv".format(self.experiment_name, self.file_name), mode="a+") as csv_file:
            self.writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(row)
            csv_file.flush()

class ExperimentBase:
    def __init__(self, conf):
        self.conf = conf

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.cuda = True
        else:
            self.device = 'cpu'
            self.cuda = False

        self.compute_experiment_name()

        self.create_experiment_folders()

        self.iteration = 0
        self.train_loader = None
        self.val_loader = None

        # TODO: How to write first line correctly?  
        self.remove_if_exists("experiments/{}/{}.csv".format(self.experiment_name, "loss"), file=True)
        self.train_writer = CSVWriter(self.experiment_name, "loss")

        self.remove_if_exists("experiments/{}/{}.csv".format(self.experiment_name, "validation"), file=True)
        self.val_writer = CSVWriter(self.experiment_name, "validation")

        self.remove_if_exists("experiments/{}/{}.csv".format(self.experiment_name, "parameters"), file=True)
        self.parameter_writer = CSVWriter(self.experiment_name, "parameters")

        for key in self.conf:
            self.parameter_writer.write([key, self.conf[key]])

        np.random.seed(self.conf["numpy_seed"])

    def preparation(self):
        raise("Preparation not implemented")

    def create_dynamic_folders(self):
        self.remove_if_exists('experiments/{}/heatmaps/{}'.format(self.experiment_name, self.iteration))
        self.create_if_not_exists('experiments/{}/heatmaps/{}'.format(self.experiment_name, self.iteration))

        self.remove_if_exists('experiments/{}/val_images/{}'.format(self.experiment_name, self.iteration))
        self.create_if_not_exists('experiments/{}/val_images/{}'.format(self.experiment_name, self.iteration))

    def compute_experiment_name(self):
        if self.conf["name"] is not None:
            self.experiment_name = self.conf["name"]
            if self.conf["project_dir"] is not None:
                self.experiment_name = self.conf["project_dir"] + "/" + self.experiment_name
        else:
            self.experiment_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def create_experiment_folders(self):
        self.create_if_not_exists("experiments")
        self.create_if_not_exists("experiments/{}".format(self.experiment_name))
        
        self.remove_if_exists('experiments/{}/weights'.format(self.experiment_name))
        self.create_if_not_exists("experiments/{}/weights".format(self.experiment_name))
        self.remove_if_exists('experiments/{}/val_images'.format(self.experiment_name))
        self.create_if_not_exists('experiments/{}/val_images'.format(self.experiment_name))
        self.remove_if_exists('experiments/{}/heatmaps'.format(self.experiment_name))
        self.create_if_not_exists('experiments/{}/heatmaps'.format(self.experiment_name))

    def create_if_not_exists(self, path):
        if not exists(path):
            makedirs(path)

    def remove_if_exists(self, path, file=False):
        if exists(path):
            if file:
                remove(path)
            else:
                rmtree(path)

    def split_indices(self, ds_length):
        number_of_datapoints = int(ds_length * self.conf["limit_data_percent"])
        indices = list(range(number_of_datapoints))
        split = int((1 - self.conf["validation_amount"]) * number_of_datapoints)

        np.random.shuffle(indices)

        train_indices = indices[:split]
        val_indices = indices[split:]
        print("Using {} training and {} validation datapoints".format(len(train_indices), len(val_indices)))
        return train_indices, val_indices

    def train(self, train_objects):
        Warning("this is an abstract class. Train needs to be implemented")
        return 0

    def evaluate(self):
        Warning("this is an abstract class. Evaluate needs to be implemented")
        return 0

    def run_experiment(self):

        self.preparation()

        for epoch in range(self.conf["nr_epochs"]):

            for train_objects in self.train_loader:

                self.train(train_objects)

                if self.iteration % self.conf["evaluate_rate"] == 0:
                    with torch.no_grad():
                        val_accuracy = self.evaluate()
                        print("-----------------------------------")
                        print("iteration {} val-accuracy {}".format(self.iteration, val_accuracy))
                        print("-----------------------------------")


class HAR_Testing_Experiment(ExperimentBase):
    def preparation(self):
        
        self.model = DeepHar(num_actions=21, use_gt=True, model_path="/Users/Matthias/code/master-thesis/code/weights_00010000").to(self.device)
        self.ds = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/")

        train_indices, val_indices = self.split_indices(len(self.ds))

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = data.DataLoader(
            self.ds,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds,
            batch_size=self.conf["batch_size"],
            sampler=val_sampler
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0002, momentum=0.98, nesterov=True)

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "accuracy"])


    def train(self, train_objects):

        frames = train_objects["frames"].to(self.device)
        actions = train_objects["action_1h"].to(self.device)

        actions = actions.unsqueeze(1)
        actions = actions.expand(-1, 4, -1)

        _, pose_predicted_actions, vis_predicted_actions, _ = self.model(frames)

        partial_loss_pose = torch.sum(categorical_cross_entropy(pose_predicted_actions, actions))
        partial_loss_action = torch.sum(categorical_cross_entropy(vis_predicted_actions, actions))
        losses = partial_loss_pose + partial_loss_action

        losses.backward()

        self.train_writer.write([self.iteration, losses])
        print(self.iteration, losses.item())

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.iteration = self.iteration + 1

        print("iteration {} train-loss {}".format(self.iteration, losses.item()))
        
    def evaluate(self):
        correct = 0
        total = 0
        for validation_objects in self.val_loader:
            frames = validation_objects["frames"].to(self.device)
            actions = validation_objects["action_1h"].to(self.device)

            ground_class = torch.argmax(actions[0])

            _, _, _, prediction = self.model(frames)

            pred_class = torch.argmax(prediction)

            total = total + 1
            if pred_class == ground_class:
                correct = correct + 1
        
        accuracy = correct / float(total)
        
        self.val_writer.write([self.iteration, accuracy])

        return accuracy


class MPIIExperiment(ExperimentBase):

    def preparation(self):
        self.ds = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=False, use_saved_tensors=self.conf["use_saved_tensors"])

        if self.conf["num_blocks"] == 1:
            self.model = Mpii_1(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 2:
            self.model = Mpii_2(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 4:
            self.model = Mpii_4(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 8:
            self.model = Mpii_8(num_context=self.conf["nr_context"]).to(self.device)        

        train_indices, val_indices = self.split_indices(len(self.ds))

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = data.DataLoader(
            self.ds,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds,
            batch_size=self.conf["batch_size"],
            sampler=val_sampler
        )

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.conf["learning_rate"])

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "pckh_0.5", "pckh_0.2"])

    def train(self, train_objects):
        images = train_objects["normalized_image"].to(self.device)
        poses = train_objects["normalized_pose"].to(self.device)

        heatmaps, output = self.model(images)

        output = output.permute(1, 0, 2, 3)

        pred_pose = output[:, :, :, 0:2]
        ground_pose = poses[:, :, 0:2]
        ground_pose = ground_pose.unsqueeze(1)
        ground_pose = ground_pose.expand(-1, self.conf["num_blocks"], -1, -1)

        pred_vis = output[:, :, :, 2]
        ground_vis = poses[:, :, 2]
        ground_vis = ground_vis.unsqueeze(1)
        ground_vis = ground_vis.expand(-1, self.conf["num_blocks"], -1)

        binary_crossentropy = nn.BCELoss()

        vis_loss = binary_crossentropy(pred_vis, ground_vis)

        pose_loss = elastic_net_loss_paper(pred_pose, ground_pose)
        loss = vis_loss * 0.01 + pose_loss

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.train_writer.write([self.iteration, loss.item()])


        self.iteration = self.iteration + 1

        print("iteration {} loss {}".format(self.iteration, loss.item()))


    def evaluate(self):
        val_accuracy_05 = []
        val_accuracy_02 = []
        for batch_idx, val_data in enumerate(self.val_loader):
            val_images = val_data["normalized_image"].to(self.device)

            heatmaps, predictions = self.model(val_images)
            predictions = predictions[-1, :, :, :].squeeze(dim=0)

            if predictions.dim() == 2:
                predictions = predictions.unsqueeze(0)

            image_number = "{}".format(int(val_data["image_path"][0].item()))
            image_name = "{}.jpg".format(image_number.zfill(9))
            image = io.imread("/data/mjakobs/data/mpii/images/{}".format(image_name))

            self.create_dynamic_folders()

            if batch_idx % 10 == 0:
                #visualize_heatmaps(heatmaps[0], val_images[0], 'experiments/{}/heatmaps/{}/{}_hm.png'.format(experiment_name, iteration, batch_idx), save=True)
                show_predictions_ontop(val_data["normalized_pose"][0], image, predictions[0], 'experiments/{}/val_images/{}/{}.png'.format(self.experiment_name, self.iteration, batch_idx), val_data["trans_matrix"][0], bbox=val_data["bbox"][0], save=True)

            scores_05, scores_02 = eval_pckh_batch(predictions, val_data["normalized_pose"], val_data["head_size"], val_data["trans_matrix"])
            val_accuracy_05.extend(scores_05)
            val_accuracy_02.extend(scores_02)


        mean_05 = np.mean(np.array(val_accuracy_05))
        mean_02 = np.mean(np.array(val_accuracy_02))

        self.val_writer.write([self.iteration, mean_05, mean_02])
        return mean_05