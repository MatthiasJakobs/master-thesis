from datetime import datetime

from shutil import rmtree

from os import makedirs, remove
from os.path import exists

import matplotlib.pyplot as plt

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
from datasets.JHMDBDataset import actions as jhmdb_actions
from deephar.models import DeepHar, Mpii_1, Mpii_2, Mpii_4, Mpii_8, TimeDistributedPoseEstimation
from deephar.utils import get_valid_joints
from deephar.measures import elastic_net_loss_paper, categorical_cross_entropy
from deephar.evaluation import *

from visualization import show_predictions_ontop, visualize_heatmaps, show_prediction_jhmbd

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

    def create_dynamic_folders(self, heatmaps=True, val_images=True):
        if heatmaps:
            self.remove_if_exists('experiments/{}/heatmaps/{}'.format(self.experiment_name, self.iteration))
            self.create_if_not_exists('experiments/{}/heatmaps/{}'.format(self.experiment_name, self.iteration))
        if val_images:
            self.remove_if_exists('experiments/{}/val_images/{}'.format(self.experiment_name, self.iteration))
            self.create_if_not_exists('experiments/{}/val_images/{}'.format(self.experiment_name, self.iteration))

    def compute_experiment_name(self):
        if self.conf["name"] is not None:
            self.experiment_name = self.conf["name"]
            if self.conf["project_dir"] != "":
                self.experiment_name = self.conf["project_dir"] + "/" + self.experiment_name
        else:
            self.experiment_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def create_experiment_folders(self, heatmaps=True, val_images=True):
        self.create_if_not_exists("experiments")
        self.create_if_not_exists("experiments/{}".format(self.experiment_name))

        self.remove_if_exists('experiments/{}/weights'.format(self.experiment_name))
        self.create_if_not_exists("experiments/{}/weights".format(self.experiment_name))

        if val_images:
            self.remove_if_exists('experiments/{}/val_images'.format(self.experiment_name))
            self.create_if_not_exists('experiments/{}/val_images'.format(self.experiment_name))

        if heatmaps:
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

    def test(self, pretrained_model=None):
        Warning("this is an abstract class. Test needs to be implemented")
        return 0

    def run_experiment(self):

        self.preparation()

        while True:

            for train_objects in self.train_loader:

                self.train(train_objects)

                if self.iteration % self.conf["evaluate_rate"] == 0:
                    with torch.no_grad():
                        val_accuracy = self.evaluate()
                        print("-----------------------------------")
                        print("iteration {} val-accuracy {}".format(self.iteration, val_accuracy))
                        print("-----------------------------------")

                if self.iteration >= self.conf["total_iterations"]:
                    print("done")
                    return


class HAR_Testing_Experiment(ExperimentBase):
    def preparation(self):

        self.model = DeepHar(num_actions=21, use_gt=True, model_path="/data/mjakobs/data/pretrained_jhmdb").to(self.device)
        self.ds_train = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=False)
        self.ds_val = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=True)

        train_sampler = SubsetRandomSampler(list(range(len(self.ds_train))))
        val_sampler = SubsetRandomSampler(list(range(len(self.ds_val))))

        self.train_loader = data.DataLoader(
            self.ds_train,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds_val,
            batch_size=self.conf["batch_size"],
            sampler=val_sampler
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.conf["learning_rate"], momentum=0.98, nesterov=True)

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "accuracy"])

        self.create_experiment_folders(heatmaps=False)


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

        self.train_writer.write([self.iteration, losses.item()])

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.iteration = self.iteration + 1

        print("iteration {} train-loss {}".format(self.iteration, losses.item()))

    def evaluate(self):
        correct = 0
        total = 0
        for batch_idx, validation_objects in enumerate(self.val_loader):
            frames = validation_objects["frames"].to(self.device)
            actions = validation_objects["action_1h"].to(self.device)

            ground_class = torch.argmax(actions, 1)

            predicted_poses, _, _, prediction = self.model(frames)

            if torch.cuda.is_available():
                frames = frames.cpu()
                actions = actions.cpu()
                predicted_poses = predicted_poses.cpu()

            if batch_idx % 10 == 0:
                self.create_dynamic_folders(heatmaps=False)
                for i in range(len(frames)):
                    for frame in range(len(frames[0])):
                        path = 'experiments/{}/val_images/{}/{}_{}_{}.png'.format(self.experiment_name, self.iteration, batch_idx, i, frame)
                        plt.imshow(frames[i, frame].reshape(255, 255, 3))

                        pred_x = predicted_poses[i, frame, :, 0]
                        pred_y = predicted_poses[i, frame, :, 1]

                        plt.scatter(x=pred_x * 255.0, y=pred_y * 255.0)
                        plt.savefig(path)
                        plt.close()

            pred_class = torch.argmax(prediction.squeeze(1), 1)

            total = total + len(pred_class)
            correct = correct + torch.sum(pred_class == ground_class).item()

        accuracy = correct / float(total)

        self.val_writer.write([self.iteration, accuracy])

        torch.save(self.model.state_dict(), "experiments/{}/weights/weights_{:08d}".format(self.experiment_name, self.iteration))

        return accuracy

class Pose_JHMDB(ExperimentBase):

    def __init__(self, conf, use_pretrained=False):
        super().__init__(conf)
        self.use_pretrained = use_pretrained

    def preparation(self):

        self.model = Mpii_4(num_context=0, standalone=True).to(self.device)
        if self.use_pretrained:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load("/data/mjakobs/data/pretrained_weights_4", map_location=self.device))
        self.model.train()

        self.ds_train = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=False)
        self.ds_val = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=True)

        train_sampler = SubsetRandomSampler(list(range(len(self.ds_train))))
        val_sampler = SubsetRandomSampler(list(range(len(self.ds_val))))

        self.train_loader = data.DataLoader(
            self.ds_train,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds_val,
            batch_size=self.conf["batch_size"],
            sampler=val_sampler
        )

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.conf["learning_rate"])

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "pckh_0.2"])

        self.create_experiment_folders()


    def train(self, train_objects):

        images = train_objects["frames"].to(self.device)
        train_poses = train_objects["poses"].to(self.device)

        images = images.contiguous().view(images.size()[0] * images.size()[1], 3, 255, 255)
        train_poses = train_poses.contiguous().view(train_poses.size()[0] * train_poses.size()[1], 16, 3)

        heatmaps, poses = self.model(images)

        poses = poses.permute(1, 0, 2, 3)

        pred_pose = poses[:, :, :, 0:2]
        ground_pose = train_poses[:, :, 0:2]
        ground_pose = ground_pose.unsqueeze(1)
        ground_pose = ground_pose.expand(-1, self.conf["num_blocks"], -1, -1)

        pred_vis = poses[:, :, :, 2]
        ground_vis = train_poses[:, :, 2]
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
        val_accuracy_02 = []

        self.create_dynamic_folders()

        for batch_idx, val_data in enumerate(self.val_loader):
            val_images = val_data["frames"].to(self.device)
            val_images = val_images.contiguous().view(val_data["frames"].size()[0] * val_data["frames"].size()[1], 3, 255, 255)

            val_poses = val_data["poses"].to(self.device)
            val_poses = val_poses.contiguous().view(val_data["poses"].size()[0] * val_data["poses"].size()[1], 16, 3)

            trans_matrices = val_data["trans_matrices"].to(self.device)
            trans_matrices = trans_matrices.contiguous().view(val_data["trans_matrices"].size()[0] * val_data["trans_matrices"].size()[1], 3, 3)

            heatmaps, predictions = self.model(val_images)
            predictions = predictions[-1, :, :, :].squeeze(dim=0)

            if predictions.dim() == 2:
                predictions = predictions.unsqueeze(0)


            # save predictions
            image = val_images[0].reshape(255, 255, 3)
            gt_poses = val_poses

            val_accuracy_02.append(eval_pcku_batch(predictions[:, :, 0:2], gt_poses[:, :, 0:2], trans_matrices))

            if batch_idx % 10 == 0:
                prediction = predictions[0, :, 0:2]
                gt_pose = gt_poses[0]
                matrix = trans_matrices[0]

                if torch.cuda.is_available():
                    image = image.cpu()
                    prediction = prediction.cpu()
                    gt_pose = gt_pose.cpu()

                path = 'experiments/{}/val_images/{}/{}.png'.format(self.experiment_name, self.iteration, batch_idx)

                show_prediction_jhmbd(image, gt_pose, prediction, matrix, path=path)

                # plt.imshow(image)
                # pred_x = prediction[:, 0]
                # pred_y = prediction[:, 1]
                # gt_x = gt_pose[:, 0]
                # gt_y = gt_pose[:, 1]
                # plt.scatter(x=pred_x, y=pred_y, c="b")
                # plt.scatter(x=gt_x, y=gt_y, c="r")
                # path = 'experiments/{}/val_images/{}/{}.png'.format(self.experiment_name, self.iteration, batch_idx)
                # plt.savefig(path)
                # plt.close()


        torch.save(self.model.state_dict(), "experiments/{}/weights/weights_{:08d}".format(self.experiment_name, self.iteration))

        mean_02 = torch.mean(torch.FloatTensor(val_accuracy_02)).item()
        self.val_writer.write([self.iteration, mean_02])
        return mean_02



class MPIIExperiment(ExperimentBase):

    def preparation(self):
        self.ds_train = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=self.conf["use_random_parameters"], use_saved_tensors=self.conf["use_saved_tensors"])
        self.ds_val = MPIIDataset("/data/mjakobs/data/mpii/", use_random_parameters=False, use_saved_tensors=self.conf["use_saved_tensors"])

        if self.conf["num_blocks"] == 1:
            self.model = Mpii_1(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 2:
            self.model = Mpii_2(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 4:
            self.model = Mpii_4(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 8:
            self.model = Mpii_8(num_context=self.conf["nr_context"]).to(self.device)

        train_indices, val_indices = self.split_indices(len(self.ds_train))

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = data.DataLoader(
            self.ds_train,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds_val,
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
        self.create_dynamic_folders()

        for batch_idx, val_data in enumerate(self.val_loader):
            val_images = val_data["normalized_image"].to(self.device)

            heatmaps, predictions = self.model(val_images)
            predictions = predictions[-1, :, :, :].squeeze(dim=0)

            if predictions.dim() == 2:
                predictions = predictions.unsqueeze(0)

            image_number = "{}".format(int(val_data["image_path"][0].item()))
            image_name = "{}.jpg".format(image_number.zfill(9))
            image = io.imread("/data/mjakobs/data/mpii/images/{}".format(image_name))

            if batch_idx % 10 == 0:
                #visualize_heatmaps(heatmaps[0], val_images[0], 'experiments/{}/heatmaps/{}/{}_hm.png'.format(experiment_name, iteration, batch_idx), save=True)
                show_predictions_ontop(val_data["normalized_pose"][0], image, predictions[0], 'experiments/{}/val_images/{}/{}.png'.format(self.experiment_name, self.iteration, batch_idx), val_data["trans_matrix"][0], bbox=val_data["bbox"][0], save=True)

            scores_05, scores_02 = eval_pckh_batch(predictions, val_data["normalized_pose"], val_data["head_size"], val_data["trans_matrix"])
            val_accuracy_05.extend(scores_05)
            val_accuracy_02.extend(scores_02)


        mean_05 = np.mean(np.array(val_accuracy_05))
        mean_02 = np.mean(np.array(val_accuracy_02))

        torch.save(self.model.state_dict(), "experiments/{}/weights/weights_{:08d}".format(self.experiment_name, self.iteration))

        self.val_writer.write([self.iteration, mean_05, mean_02])
        return mean_05
