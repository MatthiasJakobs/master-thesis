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

from sklearn.metrics import confusion_matrix

import csv
import numpy as np
import math
import pdb

from skimage import io
from skimage.transform import resize

from datasets.MPIIDataset import *
from datasets.JHMDBFragmentsDataset import JHMDBFragmentsDataset
from datasets.JHMDBDataset import actions as jhmdb_actions
from datasets.JHMDBDataset import JHMDBDataset
from datasets.MixMPIIPenn import MixMPIIPenn
from datasets.PennActionDataset import PennActionDataset
from datasets.PennActionDataset import actions as pennaction_actions
from datasets.PennActionFragmentsDataset import PennActionFragmentsDataset
from deephar.models import DeepHar, DeepHar_Smaller, Mpii_1, Mpii_2, Mpii_4, Mpii_8, TimeDistributedPoseEstimation
from deephar.blocks import Stem
from deephar.utils import get_valid_joints, get_bbox_from_pose, transform_2d_point, transform_pose
from deephar.measures import elastic_net_loss_paper, categorical_cross_entropy
from deephar.evaluation import *
from deephar.image_processing import center_crop

from visualization import show_predictions_ontop, visualize_heatmaps, show_prediction_jhmbd

class CSVWriter:
    def __init__(self, experiment_name, file_name, remove=False):
        self.experiment_name = experiment_name
        self.file_name = file_name
        self.remove = remove

        if self.remove:
            if os.path.exists("experiments/{}/{}.csv".format(self.experiment_name, self.file_name)):
                os.remove("experiments/{}/{}.csv".format(self.experiment_name, self.file_name))

    def write(self, row):
        with open("experiments/{}/{}.csv".format(self.experiment_name, self.file_name), mode="a+") as csv_file:
            self.writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(row)
            csv_file.flush()

class ExperimentBase:
    def __init__(self, conf, validate=False):
        self.conf = conf

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.cuda = True
        else:
            self.device = 'cpu'
            self.cuda = False

        self.compute_experiment_name()

        if not validate:
            self.create_experiment_folders()

        self.iteration = 0
        self.train_loader = None
        self.val_loader = None

        if not validate:
            self.remove_if_exists("experiments/{}/{}.csv".format(self.experiment_name, "loss"), file=True)
            self.remove_if_exists("experiments/{}/{}.csv".format(self.experiment_name, "validation"), file=True)
            self.remove_if_exists("experiments/{}/{}.csv".format(self.experiment_name, "parameters"), file=True)

        self.train_writer = CSVWriter(self.experiment_name, "loss")

        self.val_writer = CSVWriter(self.experiment_name, "validation")

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

    def limit_dataset(self, include_test=False):
        train_indices = []
        val_indices = []
        test_indices = []

        datasets = [self.ds_train, self.ds_val]
        if include_test:
            datasets.append(self.ds_test)

        for i, ds in enumerate(datasets):
            datapoints = int(len(ds) * self.conf["limit_data_percent"])

            indices = list(range(len(ds)))

            np.random.shuffle(indices)

            indices = indices[:datapoints]

            if i == 0:
                train_indices = indices
            if i == 1:
                val_indices = indices
            if i == 2:
                test_indices = indices

        if include_test:
            print("Using {} training, {} validation and {} test datapoints".format(len(train_indices), len(val_indices), len(test_indices)))
            return train_indices, val_indices, test_indices
        else:
            print("Using {} training and {} validation datapoints".format(len(train_indices), len(val_indices)))
            return train_indices, val_indices

    def train(self, train_objects):
        print("Train not implemented")
        return 0

    def evaluate(self):
        print("Evaluation not implemented")
        return 0

    def test(self, pretrained_model=None):
        print("Test not implemented")
        return 0

    def run_experiment(self):

        self.preparation()

        running = True

        while running:

            for train_objects in self.train_loader:
                self.train(train_objects)

                if self.iteration % self.conf["evaluate_rate"] == 0:
                    with torch.no_grad():
                        val_accuracy = self.evaluate()
                        print("-----------------------------------")
                        print("iteration {} val-accuracy {}".format(self.iteration, val_accuracy))
                        print("-----------------------------------")

                if self.iteration >= self.conf["total_iterations"]:
                    print("Done training")
                    running = False
                    break

        # test_accuracy = self.test()
        # print("Test accuracy: " + str(test_accuracy))
        return


class HAR_Testing_Experiment(ExperimentBase):
    def __init__(self, conf, start_at=None, pretrained_model=None):
        super().__init__(conf)
        self.pretrained_model = pretrained_model
        self.start_at = start_at
        print(self.start_at)

    def preparation(self, load_model=True, nr_aug=10):
        if "fine_tune" in self.conf:
            self.fine_tune = self.conf["fine_tune"]
        else:
            self.fine_tune = False

        if "use_gt_bb" in self.conf:
            self.use_gt_bb = self.conf["use_gt_bb"]
        else:
            self.use_gt_bb = False

        if "use_gt_pose" in self.conf:
            self.use_gt_pose = self.conf["use_gt_pose"]
        else:
            self.use_gt_pose = False

        if "use_timedistributed" in self.conf:
            self.use_timedistributed = self.conf["use_timedistributed"]
        else:
            self.use_timedistributed = False

        if self.start_at is not None:
            print('startat')
            print(self.start_at)
            self.model = DeepHar(num_actions=21, use_gt=False, nr_context=self.conf["nr_context"], use_timedistributed=self.use_timedistributed).to(self.device)
            self.model.load_state_dict(torch.load(self.start_at, map_location=self.device))
        else:
            if load_model:
                self.model = DeepHar(num_actions=21, use_gt=True, nr_context=self.conf["nr_context"], model_path="/data/mjakobs/data/pretrained_jhmdb", use_timedistributed=self.use_timedistributed).to(self.device)

                if self.pretrained_model is not None:
                    self.model.load_state_dict(torch.load(self.pretrained_model, map_location=self.device))
                  
            else:
                self.model = DeepHar(num_actions=21, use_gt=False, nr_context=self.conf["nr_context"], use_timedistributed=self.use_timedistributed).to(self.device)

        self.ds_train = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=False, use_random_parameters=True, augmentation_amount=nr_aug, use_gt_bb=self.use_gt_bb)
        self.ds_val = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=True, use_gt_bb=self.use_gt_bb)
        self.ds_test = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=False)

        print("Number of augmentations", str(nr_aug))
        train_indices, val_indices, test_indices = self.limit_dataset(include_test=True)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        self.train_loader = data.DataLoader(
            self.ds_train,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds_val,
            batch_size=self.conf["val_batch_size"],
            sampler=val_sampler
        )

        self.test_loader = data.DataLoader(
            self.ds_test,
            batch_size=1,
            sampler=test_sampler
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.conf["learning_rate"], momentum=0.98, nesterov=True)

        if "lr_milestones" in self.conf:
            milestones = self.conf["lr_milestones"]
        else:
            milestones = [20000000] # basically, never use lr scheduler

        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "accuracy", "pose_model_accuracy", "visual_model_accuracy"])

        self.create_experiment_folders(heatmaps=False)

        self.shrunk = False

        self.train_accuracy_writer = CSVWriter(self.experiment_name, "train_accuracy", remove=True)
        self.train_accuracy_writer.write(["iteration", "action_accuracy", "pose_accuracy"])

        self.pose_train_accuracies = []
        self.action_train_accuracies = []


    def train(self, train_objects):
        self.model.train()

        self.optimizer.zero_grad()

        batch_size = len(train_objects["frames"])
        batch_loss = 0

        if self.use_timedistributed:
            frames = train_objects["frames"].to(self.device)
            actions = train_objects["action_1h"].to(self.device)
            ground_poses = train_objects["poses"].to(self.device)
            trans_matrices = train_objects["trans_matrices"]
            actions_1h = actions.clone()

            #frames = frames.contiguous().view(batch_size * frames.size()[1], 3, 255, 255)
            #actions = actions.contiguous().view(batch_size * actions.size()[1], 21)
            actions = actions.unsqueeze(1)
            actions = actions.expand(-1, 4, -1)

            if self.use_gt_pose:
                gt_pose = train_objects["poses"].to(self.device)
                gt_pose = gt_pose.contiguous().view(batch_size * gt_pose.size()[1], 16, 3)
            else:
                gt_pose = None

            if "start_finetuning" in self.conf and self.iteration < self.conf["start_finetuning"]:
                predicted_poses, _, pose_predicted_actions, vis_predicted_actions, prediction = self.model(frames, finetune=False, gt_pose=gt_pose)
            else:
                if self.fine_tune:
                    print("FINETUNE")
                predicted_poses, _, pose_predicted_actions, vis_predicted_actions, prediction = self.model(frames, finetune=self.fine_tune, gt_pose=gt_pose)
            
            partial_loss_pose = torch.sum(categorical_cross_entropy(pose_predicted_actions, actions))
            partial_loss_action = torch.sum(categorical_cross_entropy(vis_predicted_actions, actions))
            losses = partial_loss_pose + partial_loss_action

            pred_pose = predicted_poses[:, :, :, :, 0:2]
            ground_pose = ground_poses[:, :, :, 0:2]
            ground_pose = ground_pose.unsqueeze(2)
            ground_pose = ground_pose.expand(-1, -1, self.conf["num_blocks"], -1, -1)

            pred_vis = predicted_poses[:, :, :, :, 2]
            ground_vis = ground_poses[:, :, :, 2]
            ground_vis = ground_vis.unsqueeze(2)
            ground_vis = ground_vis.expand(-1, -1, self.conf["num_blocks"], -1)

            bboxes = train_objects["bbox"]
            bboxes = bboxes.contiguous().view(batch_size * 16, 4)

            distance_meassures = torch.FloatTensor(len(bboxes))

            for i in range(len(bboxes)):
                width = torch.abs(bboxes[i, 0] - bboxes[i, 2])
                height = torch.abs(bboxes[i, 1] - bboxes[i, 3])

                distance_meassures[i] = torch.max(width, height).item()


            pred_pose = pred_pose.contiguous().view(batch_size * pred_pose.size()[1], 4, 16, 2)
            ground_pose = ground_pose.contiguous().view(batch_size * ground_pose.size()[1], 4, 16, 2)
            trans_matrices = trans_matrices.contiguous().view(batch_size * trans_matrices.size()[1], 3, 3)
            estimates = eval_pck_batch(pred_pose[:, -1, :, 0:2], ground_pose[:, -1, :, 0:2], trans_matrices, distance_meassures)
            estimates = np.array(list(map(lambda x: x.item(), estimates)))
            estimates = estimates[np.logical_not(np.isnan(estimates))]
            self.pose_train_accuracies.append(torch.mean(torch.Tensor(estimates)).item())
            if np.isnan(np.array(self.pose_train_accuracies)).any():
                print("NAN IN GLOBAL LIST")


            predicted_class = torch.argmax(prediction.squeeze(1), 1)
            ground_class = torch.argmax(actions_1h, 1)
            self.action_train_accuracies.append(torch.sum(predicted_class == ground_class).item() / batch_size)

            del frames
            del actions

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            losses.backward()

            batch_loss = float(losses)
            self.optimizer.step()

        else:
            for i in range(batch_size):
                frames = train_objects["frames"][i].to(self.device)
                actions = train_objects["action_1h"][i].to(self.device)
                
                if self.use_gt_pose:
                    gt_pose = train_objects["poses"][i].to(self.device)
                else:
                    gt_pose = None

                actions = actions.unsqueeze(0)
                actions = actions.expand(4, -1)
                actions = actions.unsqueeze(0)

                if "start_finetuning" in self.conf and self.iteration < self.conf["start_finetuning"]:
                    _, _, pose_predicted_actions, vis_predicted_actions, _ = self.model(frames, finetune=False, gt_pose=gt_pose)
                else:
                    _, _, pose_predicted_actions, vis_predicted_actions, _ = self.model(frames, finetune=self.fine_tune, gt_pose=gt_pose)

                partial_loss_pose = torch.sum(categorical_cross_entropy(pose_predicted_actions, actions))
                partial_loss_action = torch.sum(categorical_cross_entropy(vis_predicted_actions, actions))
                losses = partial_loss_pose + partial_loss_action

                del frames
                del actions

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                losses = losses / batch_size
                losses.backward()

                batch_loss += float(losses)

            self.optimizer.step()

        self.train_writer.write([self.iteration, batch_loss])
        self.lr_scheduler.step()

        self.iteration = self.iteration + 1

        print("iteration {} train-loss {}".format(self.iteration, batch_loss))

    def evaluate(self):
        self.model.eval()
        mean_action = torch.mean(torch.Tensor(self.action_train_accuracies)).item()
        mean_pose = torch.mean(torch.Tensor(self.pose_train_accuracies)).item()
        self.train_accuracy_writer.write([self.iteration, mean_action, mean_pose])

        self.action_train_accuracies = []
        self.pose_train_accuracies = []

        with torch.no_grad():
            correct = 0
            total = 0
            pose_model_correct = 0
            visual_model_correct = 0
            for batch_idx, validation_objects in enumerate(self.val_loader):
                frames = validation_objects["frames"].to(self.device)
                actions = validation_objects["action_1h"].to(self.device)
                
                if self.use_gt_pose:
                    gt_pose = validation_objects["poses"].to(self.device)
                    gt_pose = gt_pose.squeeze(0)
                else:
                    gt_pose = None
                
                ground_class = torch.argmax(actions, 1)

                assert len(frames) == 1
                if not self.use_timedistributed:
                    frames = frames.squeeze(0)

                _, predicted_poses, pose_model_pred, visual_model_pred, prediction = self.model(frames, gt_pose=gt_pose)

                pose_model_correct = pose_model_correct + (torch.argmax(pose_model_pred[0][-1]) == ground_class).item()
                visual_model_correct = visual_model_correct + (torch.argmax(visual_model_pred[0][-1]) == ground_class).item()

                if torch.cuda.is_available():
                    frames = frames.cpu()
                    actions = actions.cpu()
                    predicted_poses = predicted_poses.cpu()

                if batch_idx % 10 == 0:
                    self.create_dynamic_folders(heatmaps=False)
                    for frame in range(len(frames)):
                        path = 'experiments/{}/val_images/{}/{}_{}.png'.format(self.experiment_name, self.iteration, batch_idx, frame)
                        if len(frames) == 1:
                            frames = frames.squeeze(0)

                        plt.imshow((frames[frame].permute(1,2,0) + 1) / 2.0)

                        pred_x = predicted_poses[frame, :, 0]
                        pred_y = predicted_poses[frame, :, 1]

                        plt.scatter(x=pred_x * 255.0, y=pred_y * 255.0, c="#FF00FF")
                        plt.savefig(path)
                        plt.close()

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

            self.val_writer.write([self.iteration, accuracy, pose_model_accuracy, visual_model_accuracy])

            torch.save(self.model.state_dict(), "experiments/{}/weights/weights_{:08d}".format(self.experiment_name, self.iteration))
            torch.save(self.model.pose_estimator.state_dict(), "experiments/{}/weights/pe_weights_{:08d}".format(self.experiment_name, self.iteration))

            return accuracy

    def test(self, pretrained_model=None):
        with torch.no_grad():
            if pretrained_model is not None:
                self.preparation(load_model=False)
                self.model.load_state_dict(torch.load(pretrained_model, map_location=self.device))

            self.model.eval()

            length = len(self.ds_test)
            current = 1

            correct_single = 0
            correct_multi = 0
            total = 0
            accuracies_single = []
            accuracies_multi = []
            conf_x = []
            conf_y = []
            for batch_idx, test_objects in enumerate(self.test_loader):
                frames = test_objects["normalized_frames"].to(self.device)
                actions = test_objects["action_1h"].to(self.device)
                sequence_length = test_objects["sequence_length"].to(self.device).item()
                padding = int((sequence_length - 16) / 2.0)

                ground_class = torch.argmax(actions, 1)
                single_clip = frames[:, padding:(16 + padding)].squeeze(0)
                assert len(single_clip) == 16

                spacing = 8
                nr_multiple_clips = int((sequence_length - 16) / spacing) + 1

                single_clip = single_clip.unsqueeze(0).to(self.device)
                _, _, _, _, single_result = self.model(single_clip)

                pred_class_multi = torch.IntTensor(nr_multiple_clips).to(self.device)
                for i in range(nr_multiple_clips):
                    multi_clip = frames[0, i * spacing : i * spacing + 16].unsqueeze(0).to(self.device)
                    _, _, _, _, estimated_class = self.model(multi_clip)
                    pred_class_multi[i] = torch.argmax(estimated_class.squeeze(1), 1)

                pred_class_single = torch.argmax(single_result.squeeze(1), 1)
                conf_x.append(pred_class_single.item())
                conf_y.append(ground_class.item())

                correct_single = correct_single + (pred_class_single == ground_class).item()

                ground_class = ground_class.expand(nr_multiple_clips).long()

                majority_correct = torch.sum((pred_class_multi.long() == ground_class).float()) >= (nr_multiple_clips / 2.0)

                correct_multi = correct_multi + majority_correct.int().item()
                total = total + 1

                accuracy_single = correct_single / float(total)
                accuracy_multi = correct_multi / float(total)
                accuracies_single.append(accuracy_single)
                accuracies_multi.append(accuracy_multi)
                
                current = current + 1

            cm = confusion_matrix(np.array(conf_y), np.array(conf_x))
            np.save("experiments/{}/cm.np".format(self.experiment_name), cm)
            mean_acc_single = torch.mean(torch.Tensor(accuracies_single)).item()
            mean_acc_multi = torch.mean(torch.Tensor(accuracies_multi)).item()
            return mean_acc_single, mean_acc_multi

class HAR_PennAction(HAR_Testing_Experiment):

    def preparation(self):
        if "fine_tune" in self.conf:
            self.fine_tune = self.conf["fine_tune"]
        else:
            self.fine_tune = False

        if "use_gt_bb" in self.conf:
            self.use_gt_bb = self.conf["use_gt_bb"]
        else:
            self.use_gt_bb = False

        if "use_timedistributed" in self.conf:
            self.use_timedistributed = self.conf["use_timedistributed"]
        else:
            self.use_timedistributed = False

        self.use_gt_pose = False

        self.model = DeepHar(num_actions=15, use_gt=True, nr_context=self.conf["nr_context"], model_path="/data/mjakobs/data/pretrained_mixed_pose", use_timedistributed=self.use_timedistributed).to(self.device)

        if self.pretrained_model is not None:
            self.model.load_state_dict(torch.load(self.pretrained_model, map_location=self.device))

        self.ds_train = PennActionFragmentsDataset("/data/mjakobs/data/pennaction_fragments/", train=True, val=False, use_random_parameters=True, augmentation_amount=10, use_gt_bb=self.use_gt_bb)
        self.ds_val = PennActionFragmentsDataset("/data/mjakobs/data/pennaction_fragments/", train=True, val=True, use_gt_bb=self.use_gt_bb)
        self.ds_test = PennActionDataset("/data/mjakobs/data/pennaction/", train=False)

        train_indices, val_indices, test_indices = self.limit_dataset(include_test=True)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        self.train_loader = data.DataLoader(
            self.ds_train,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds_val,
            batch_size=1,
            sampler=val_sampler
        )

        self.test_loader = data.DataLoader(
            self.ds_test,
            batch_size=1,
            sampler=test_sampler
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.conf["learning_rate"], momentum=0.98, nesterov=True)

        if "lr_milestones" in self.conf:
            milestones = self.conf["lr_milestones"]
        else:
            milestones = [20000000] # basically, never use lr scheduler

        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "accuracy", "pose_model_accuracy", "visual_model_accuracy"])

        self.create_experiment_folders(heatmaps=False)

        self.train_accuracy_writer = CSVWriter(self.experiment_name, "train_accuracy", remove=True)
        self.train_accuracy_writer.write(["iteration", "action_accuracy", "pose_accuracy"])

        self.pose_train_accuracies = []
        self.action_train_accuracies = []


class HAR_E2E(HAR_Testing_Experiment):

    def __init__(self, conf, small_model=False):
        super().__init__(conf)
        self.small_model = small_model
        self.nr_intermediate = 4
        if self.small_model:
            self.nr_intermediate = 2

    def preparation(self):
        super().preparation(load_model=False, nr_aug=10)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.conf["learning_rate"], weight_decay=0.9, momentum=0.98, nesterov=True)
        if self.small_model:
            self.model = DeepHar_Smaller(num_actions=21, use_gt=False, nr_context=self.conf["nr_context"], use_timedistributed=self.use_timedistributed).to(self.device)

    def train(self, train_objects):
        self.model.train()
        self.optimizer.zero_grad()


        batch_size = len(train_objects["frames"])
        batch_loss = 0

        if self.use_timedistributed:
            frames = train_objects["frames"].to(self.device)
            actions_1h = train_objects["action_1h"].to(self.device)
            ground_poses = train_objects["poses"].to(self.device)
            trans_matrices = train_objects["trans_matrices"]

            actions = actions_1h.unsqueeze(1)
            actions = actions.expand(-1, self.nr_intermediate, -1)

            predicted_poses, _, pose_predicted_actions, vis_predicted_actions, prediction = self.model(frames, finetune=True)

            predicted_class = torch.argmax(prediction.squeeze(1), 1)
            ground_class = torch.argmax(actions_1h, 1)
            self.action_train_accuracies.append(torch.sum(predicted_class == ground_class).item() / batch_size)

            partial_loss_pose = torch.sum(categorical_cross_entropy(pose_predicted_actions, actions))
            partial_loss_action = torch.sum(categorical_cross_entropy(vis_predicted_actions, actions))
            har_loss = partial_loss_pose + partial_loss_action

            pred_pose = predicted_poses[:, :, :, :, 0:2]
            ground_pose = ground_poses[:, :, :, 0:2]
            ground_pose = ground_pose.unsqueeze(2)
            ground_pose = ground_pose.expand(-1, -1, self.conf["num_blocks"], -1, -1)

            pred_vis = predicted_poses[:, :, :, :, 2]
            ground_vis = ground_poses[:, :, :, 2]
            ground_vis = ground_vis.unsqueeze(2)
            ground_vis = ground_vis.expand(-1, -1, self.conf["num_blocks"], -1)

            del predicted_poses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            binary_crossentropy = nn.BCELoss()
            vis_loss = binary_crossentropy(pred_vis, ground_vis)

            pred_pose = pred_pose.contiguous().view(batch_size * 16, self.nr_intermediate, 16, 2)
            ground_pose = ground_pose.contiguous().view(batch_size * 16, self.nr_intermediate, 16, 2)
            trans_matrices = trans_matrices.contiguous().view(batch_size * 16, 3, 3).to(self.device)

            pose_loss = elastic_net_loss_paper(pred_pose, ground_pose)
            pose_loss = vis_loss * 0.01 + pose_loss

            loss = pose_loss + har_loss

            bboxes = train_objects["bbox"]
            bboxes = bboxes.contiguous().view(batch_size * 16, 4)

            distance_meassures = torch.FloatTensor(len(bboxes))

            for i in range(len(bboxes)):
                width = torch.abs(bboxes[i, 0] - bboxes[i, 2])
                height = torch.abs(bboxes[i, 1] - bboxes[i, 3])

                distance_meassures[i] = torch.max(width, height).item()

            self.pose_train_accuracies.append(torch.mean(torch.Tensor(eval_pck_batch(pred_pose[:, -1, :, 0:2], ground_pose[:, -1, :, 0:2], trans_matrices, distance_meassures))).item())

            del actions
            del ground_poses
            del pose_predicted_actions
            del vis_predicted_actions
            del pred_pose
            del ground_pose
            del pred_vis
            del ground_vis

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            loss.backward()
            batch_loss = float(loss)
        
        else:
            for i in range(batch_size):
                frames = train_objects["frames"][i].to(self.device)

                predicted_poses, _, pose_predicted_actions, vis_predicted_actions, _ = self.model(frames, finetune=True)
                del frames
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                actions = train_objects["action_1h"][i].to(self.device)
                ground_poses = train_objects["poses"][i].to(self.device)

                actions = actions.unsqueeze(0)
                actions = actions.expand(self.conf["num_blocks"], -1)
                actions = actions.unsqueeze(0)

                partial_loss_pose = torch.sum(categorical_cross_entropy(pose_predicted_actions, actions))
                partial_loss_action = torch.sum(categorical_cross_entropy(vis_predicted_actions, actions))

                har_loss = partial_loss_pose + partial_loss_action

                pred_pose = predicted_poses[:, :, :, 0:2]
                ground_pose = ground_poses[:, :, 0:2]
                ground_pose = ground_pose.unsqueeze(1)
                ground_pose = ground_pose.expand(-1, self.conf["num_blocks"], -1, -1)

                pred_vis = predicted_poses[:, :, :, 2]
                ground_vis = ground_poses[:, :, 2]
                ground_vis = ground_vis.unsqueeze(1)
                ground_vis = ground_vis.expand(-1, self.conf["num_blocks"], -1)

                del predicted_poses
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                binary_crossentropy = nn.BCELoss()

                # pred_pose = pred_pose.contiguous().view(batch_size * 16, self.conf["num_blocks"], 16, 2)
                # ground_poses = ground_poses.contiguous().view(batch_size * 16, self.conf["num_blocks"], 16, 2)
                # pred_vis = pred_vis.contiguous().view(batch_size * 16, self.conf["num_blocks"], 16)
                # ground_vis = ground_vis.contiguous().view(batch_size * 16, self.conf["num_blocks"], 16)

                vis_loss = binary_crossentropy(pred_vis, ground_vis)

                pred_pose = pred_pose.contiguous().view(batch_size * 16, 4, 16, 2)
                ground_pose = ground_pose.contiguous().view(batch_size * 16, 4, 16, 2)

                pose_loss = elastic_net_loss_paper(pred_pose, ground_pose)
                pose_loss = vis_loss * 0.01 + pose_loss

                loss = pose_loss + har_loss

                del actions
                del ground_poses
                del pose_predicted_actions
                del vis_predicted_actions
                del pred_pose
                del ground_pose
                del pred_vis
                del ground_vis

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                loss.backward()
                batch_loss = batch_loss + float(loss)

        self.train_writer.write([self.iteration, batch_loss])

        self.optimizer.step()

        self.iteration = self.iteration + 1

        print("iteration {} train-loss {}".format(self.iteration, batch_loss))

    def evaluate(self):
        mean_action = torch.mean(torch.Tensor(self.action_train_accuracies)).item()
        mean_pose = torch.mean(torch.Tensor(self.pose_train_accuracies)).item()
        self.train_accuracy_writer.write([self.iteration, mean_action, mean_pose])

        self.action_train_accuracies = []
        self.pose_train_accuracies = []

        return super().evaluate()

    def test(self, pretrained_model=None):
        with torch.no_grad():
            test_ds = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=False, use_gt_bb=True)
            if pretrained_model is not None:
                self.preparation()
                self.model.load_state_dict(torch.load(pretrained_model, map_location=self.device))
    
            self.model.eval()

            correct_single = 0
            correct_multi = 0
            total = 0
            accuracies_single = []
            accuracies_multi = []
            conf_x = []
            conf_y = []
            pck_bb_02 = []
            pck_bb_01 = []
            pck_upper_02 = []
            per_joint_accuracy = np.zeros(16)
            number_valids = np.zeros(16)

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

                spacing = 8
                nr_multiple_clips = int((sequence_length - 16) / spacing) + 1

                single_clip = single_clip.unsqueeze(0).to(self.device)
                _, predicted_poses, _, _, single_result = self.model(single_clip)

                pred_class_multi = torch.IntTensor(nr_multiple_clips).to(self.device)
                for i in range(nr_multiple_clips):
                    multi_clip = frames[i * spacing : i * spacing + 16].unsqueeze(0).to(self.device)
                    _, _, _, _, estimated_class = self.model(multi_clip)
                    pred_class_multi[i] = torch.argmax(estimated_class.squeeze(1), 1)

                pred_class_single = torch.argmax(single_result.squeeze(1))
                conf_x.append(pred_class_single.item())
                conf_y.append(ground_class.item())

                correct_single = correct_single + (pred_class_single == ground_class).item()

                ground_class = ground_class.expand(nr_multiple_clips).long()

                majority_correct = torch.sum((pred_class_multi.long() == ground_class).float()) >= (nr_multiple_clips / 2.0)

                correct_multi = correct_multi + majority_correct.int().item()
                total = total + 1

                accuracy_single = correct_single / float(total)
                accuracy_multi = correct_multi / float(total)
                accuracies_single.append(accuracy_single)
                accuracies_multi.append(accuracy_multi)
                
                predicted_poses = predicted_poses.squeeze(0)

                distance_meassures = torch.FloatTensor(len(bboxes))

                for i in range(len(bboxes)):
                    width = torch.abs(bboxes[i, 0] - bboxes[i, 2])
                    height = torch.abs(bboxes[i, 1] - bboxes[i, 3])

                    distance_meassures[i] = torch.max(width, height).item()

                pck_bb_02.append(eval_pck_batch(predicted_poses[:, :, 0:2], ground_poses[:, :, 0:2], trans_matrices, distance_meassures))
                pck_bb_01.append(eval_pck_batch(predicted_poses[:, :, 0:2], ground_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.1))
                pck_upper_02.append(eval_pcku_batch(predicted_poses[:, :, 0:2], ground_poses[:, :, 0:2], trans_matrices))

                matches, valids = eval_pck_batch(predicted_poses[:, :, 0:2], ground_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.1, return_perjoint=True)

                number_valids = number_valids + np.array(valids[0])
                for u in range(16):
                    if valids[i][u]:
                        per_joint_accuracy[u] = per_joint_accuracy[u] + matches[0][u]


            cm = confusion_matrix(np.array(conf_y), np.array(conf_x))
            np.save("experiments/{}/cm.np".format(self.experiment_name), cm)
            mean_acc_single = torch.mean(torch.Tensor(accuracies_single)).item()
            mean_acc_multi = torch.mean(torch.Tensor(accuracies_multi)).item()
            mean_bb_02 = torch.mean(torch.Tensor(pck_bb_02)).item()
            mean_bb_01 = torch.mean(torch.Tensor(pck_bb_01)).item()
            mean_upper_02 = torch.mean(torch.Tensor(pck_upper_02)).item()

            number_valids[7] = 1 # joint never visible
            print(per_joint_accuracy / number_valids)

            print("mean_acc_single, mean_acc_multi, mean_bb_02, mean_bb_01, mean_upper_02")
            return mean_acc_single, mean_acc_multi, mean_bb_02, mean_bb_01, mean_upper_02


class Pose_JHMDB(ExperimentBase):

    def __init__(self, conf, validate=False, use_pretrained=False, pretrained=None, nr_aug=3, use_flip=True):
        super().__init__(conf, validate=validate)
        self.use_pretrained = use_pretrained
        self.nr_aug = nr_aug
        self.pretrained_model = pretrained

    def preparation(self):

        nr_blocks = self.conf["num_blocks"]
        context = self.conf["nr_context"]

        if nr_blocks == 2:
            self.model = Mpii_2(num_context=context).to(self.device)
        if nr_blocks == 4:
            self.model = Mpii_4(num_context=context).to(self.device)
        if nr_blocks == 8:
            self.model = Mpii_8(num_context=context).to(self.device)

        if self.use_pretrained:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(self.pretrained_model, map_location=self.device))
        self.model.train()

        if "use_random" in self.conf and self.conf["use_random"]:
            self.ds_train = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=False, augmentation_amount=self.nr_aug, use_random_parameters=True, use_gt_bb=True)
        else:
            self.ds_train = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=False, augmentation_amount=self.nr_aug, use_random_parameters=False, use_gt_bb=True)


        self.ds_val = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=True, val=True, use_gt_bb=True)
        self.ds_test = JHMDBFragmentsDataset("/data/mjakobs/data/jhmdb_fragments/", train=False, use_gt_bb=True)

        train_indices, val_indices, test_indices = self.limit_dataset(include_test=True)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        self.train_loader = data.DataLoader(
            self.ds_train,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds_val,
            batch_size=self.conf["val_batch_size"],
            sampler=val_sampler
        )

        self.test_loader = data.DataLoader(
            self.ds_test,
            batch_size=self.conf["val_batch_size"],
            sampler=test_sampler
        )

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.conf["learning_rate"])

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "pck_bb_0.2", "pck_upper_0.2"])

        self.create_experiment_folders()


    def train(self, train_objects):

        self.model.train()
        images = train_objects["frames"]
        train_poses = train_objects["poses"].to(self.device)

        images = images.contiguous().view(images.size()[0] * images.size()[1], 3, 255, 255)
        train_poses = train_poses.contiguous().view(train_poses.size()[0] * train_poses.size()[1], 16, 3)

        images = images.to(self.device)

        poses, _, _, _ = self.model(images)

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

        if math.isnan(vis_loss * 0.01 + pose_loss):
            print("is nan")
            pdb.set_trace()

        loss = vis_loss * 0.01 + pose_loss

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.train_writer.write([self.iteration, loss.item()])
        self.iteration = self.iteration + 1

        print("iteration {} loss {}".format(self.iteration, loss.item()))

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            pck_bb_02 = []
            pck_upper_02 = []

            self.create_dynamic_folders()

            for batch_idx, val_data in enumerate(self.val_loader):
                val_images = val_data["frames"].to(self.device)
                val_images = val_images.contiguous().view(val_data["frames"].size()[0] * val_data["frames"].size()[1], 3, 255, 255)

                val_poses = val_data["poses"].to(self.device)
                val_poses = val_poses.contiguous().view(val_data["poses"].size()[0] * val_data["poses"].size()[1], 16, 3)

                trans_matrices = val_data["trans_matrices"].to(self.device)
                trans_matrices = trans_matrices.contiguous().view(val_data["trans_matrices"].size()[0] * val_data["trans_matrices"].size()[1], 3, 3)

                _, predictions, _, _ = self.model(val_images)
                predictions = predictions.squeeze(dim=0)

                if predictions.dim() == 2:
                    predictions = predictions.unsqueeze(0)


                # save predictions
                image = val_images[0].permute(1, 2, 0)
                gt_poses = val_poses

                # get distance meassures
                bboxes = val_data["bbox"]
                bboxes = bboxes.contiguous().view(val_data["bbox"].size()[0] * val_data["bbox"].size()[1], 4)

                distance_meassures = torch.FloatTensor(len(bboxes))

                for i in range(len(bboxes)):
                    width = torch.abs(bboxes[i, 0] - bboxes[i, 2])
                    height = torch.abs(bboxes[i, 1] - bboxes[i, 3])

                    distance_meassures[i] = torch.max(width, height).item()

                pck_bb_02.append(eval_pck_batch(predictions[:, :, 0:2], gt_poses[:, :, 0:2], trans_matrices, distance_meassures))
                pck_upper_02.append(eval_pcku_batch(predictions[:, :, 0:2], gt_poses[:, :, 0:2], trans_matrices))

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

        mean_bb_02 = torch.mean(torch.FloatTensor(pck_bb_02)).item()
        mean_upper_02 = torch.mean(torch.FloatTensor(pck_upper_02)).item()
        self.val_writer.write([self.iteration, mean_bb_02, mean_upper_02])
        return mean_bb_02

    def test(self, pretrained_model=None, refine_bounding_box=False):
        with torch.no_grad():
            if pretrained_model is not None:
                self.preparation()
                self.model.load_state_dict(torch.load(pretrained_model, map_location=self.device))

            pck_bb_02 = []
            pck_bb_01 = []
            pck_upper_02 = []

            per_joint_accuracy = np.zeros(16)
            number_valids = np.zeros(16)

            self.model.eval()
            for batch_idx, test_data in enumerate(self.test_loader):

                test_images = test_data["frames"].to(self.device)
                batch_size = len(test_images)
                assert batch_size == 1
                test_images = test_images.squeeze(0)
                
                #test_images = test_images.contiguous().view(test_data["frames"].size()[0] * test_data["frames"].size()[1], 3, 255, 255)

                test_poses = test_data["poses"].to(self.device).squeeze(0)
                #test_poses = test_poses.contiguous().view(test_data["poses"].size()[0] * test_data["poses"].size()[1], 16, 3)

                trans_matrices = test_data["trans_matrices"].clone().to(self.device).squeeze(0)
                #trans_matrices = trans_matrices.contiguous().view(test_data["trans_matrices"].size()[0] * test_data["trans_matrices"].size()[1], 3, 3)

                original_window_sizes = test_data["original_window_size"].squeeze(0)
                #original_window_sizes = original_window_sizes.contiguous().view(test_data["original_window_size"].size()[0] * test_data["original_window_size"].size()[1], 2)

                distance_meassures = torch.FloatTensor(len(original_window_sizes))

                for i in range(len(original_window_sizes)):
                    distance_meassures[i] = original_window_sizes[i][0]

                # initial pose estimation to get more refined bounding box
                _, predictions, _, _ = self.model(test_images)
                predictions = predictions.squeeze(dim=0)

                if predictions.dim() == 2:
                    predictions = predictions.unsqueeze(0)

                if refine_bounding_box:
                    side_lenghts = []
                    for idx, prediction in enumerate(predictions):
                        prediction[:, 0:2] = prediction[:, 0:2] * 255.0
                        frame = test_images[idx]

                        bbox_parameter = get_bbox_from_pose(prediction, bbox_offset=30) # TODO: change in other places

                        #prediction[:, 0:2] = prediction[:, 0:2] / 255.0
                        center = bbox_parameter["original_center"]
                        window_size = bbox_parameter["original_window_size"] + 1
                        side_lenghts.append(window_size)

                        new_matrix = torch.eye(3)
                        matrix, frame = center_crop(frame.permute(1, 2, 0).cpu().numpy(), center, window_size, new_matrix)

                        frame = torch.from_numpy(resize(frame, (255, 255), preserve_range=True)).permute(2, 0, 1)

                        trans_matrices[idx] = torch.from_numpy(matrix).clone()
                        test_images[idx] = frame

                    _, predictions, _, _ = self.model(test_images)
                    predictions = predictions.squeeze(dim=0)

                    for idx, prediction in enumerate(predictions):
                        coordinates = prediction[:, 0:2] * float(side_lenghts[idx][0].item())
                        back_transformed = transform_pose(trans_matrices[idx], coordinates, inverse=True)
                        predictions[idx, :, 0:2] = torch.from_numpy(back_transformed) / 255.0

                    trans_matrices = test_data["trans_matrices"].clone().to(self.device)
                    trans_matrices = trans_matrices.contiguous().view(test_data["trans_matrices"].size()[0] * test_data["trans_matrices"].size()[1], 3, 3)

                try:
                    pck_bb_02.extend(eval_pck_batch(predictions[:, :, 0:2], test_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.2))
                    matches, valids = eval_pck_batch(predictions[:, :, 0:2], test_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.1, return_perjoint=True)

                    for i in range(batch_size):
                        number_valids = number_valids + np.array(valids[i])
                        for u in range(16):
                            if valids[i][u]:
                                per_joint_accuracy[u] = per_joint_accuracy[u] + matches[i][u]

                    pck_bb_01.extend(eval_pck_batch(predictions[:, :, 0:2], test_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.1))
                    pck_upper_02.extend(eval_pcku_batch(predictions[:, :, 0:2], test_poses[:, :, 0:2], trans_matrices))
                except np.linalg.linalg.LinAlgError:
                    print("hello")

                if batch_idx % 10 == 0:
                    image = test_images[0].permute(1, 2, 0)
                    prediction = predictions[0, :, 0:2]
                    test_poses = test_poses[0]
                    matrix = trans_matrices[0]

                    if torch.cuda.is_available():
                        image = image.cpu()
                        prediction = prediction.cpu()
                        test_poses = test_poses.cpu()

                    qualifier = "not_refined"
                    if refine_bounding_box:
                        qualifier = "refined"

                    folder_path = 'experiments/{}/test_images/{}_{}'.format(self.experiment_name, self.iteration, qualifier)
                    if not exists(folder_path):
                        makedirs(folder_path)
                    path = 'experiments/{}/test_images/{}/{}.png'.format(self.experiment_name, self.iteration, batch_idx)

                    show_prediction_jhmbd(image, test_poses, prediction, matrix, path=path)

            number_valids[7] = 1 # joint never visible
            print(per_joint_accuracy / number_valids)
            bb_mean = torch.mean(torch.FloatTensor(pck_bb_02)).item()
            bb_01_mean = torch.mean(torch.FloatTensor(pck_bb_01)).item()
            upper_mean = torch.mean(torch.FloatTensor(pck_upper_02)).item()
            print("bb@02, bb@01, upper")
            return [bb_mean, bb_01_mean, upper_mean]

class Pose_Mixed(ExperimentBase):

    def __init__(self, conf, validate=False):
        super().__init__(conf, validate=validate)

        self.train_accuracy_writer = CSVWriter(self.experiment_name, "train_accuracy", remove=True)
        self.train_accuracy_writer.write(["iteration", "pose_accuracy"])

        self.pose_train_accuracies = []

    def preparation(self):

        nr_blocks = self.conf["num_blocks"]
        context = self.conf["nr_context"]

        if nr_blocks == 2:
            self.model = Mpii_2(num_context=context).to(self.device)
        if nr_blocks == 4:
            self.model = Mpii_4(num_context=context).to(self.device)
        if nr_blocks == 8:
            self.model = Mpii_8(num_context=context).to(self.device)

        self.model.train()

        self.ds_train = MixMPIIPenn(train=True, val=False, use_gt_bb=True)
        self.ds_val = PennActionDataset("/data/mjakobs/data/pennaction/", train=True, val=True, use_gt_bb=True, use_saved_tensors=True)

        train_indices, val_indices = self.limit_dataset(include_test=False)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = data.DataLoader(
            self.ds_train,
            batch_size=self.conf["batch_size"],
            sampler=train_sampler
        )

        self.val_loader = data.DataLoader(
            self.ds_val,
            batch_size=1,
            sampler=val_sampler
        )

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.conf["learning_rate"])

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "pck_bb_0.2", "pck_bb_0.1", "pck_upper_0.2"])

        self.create_experiment_folders()


    def train(self, train_objects):

        self.model.train()
        images = train_objects["normalized_image"].to(self.device)
        poses = train_objects["normalized_pose"].to(self.device)
        trans_matrices = train_objects["trans_matrix"].to(self.device)

        predictions, _, _, _ = self.model(images)

        predictions = predictions.permute(1, 0, 2, 3)

        pred_pose = predictions[:, :, :, 0:2]
        ground_pose = poses[:, :, 0:2]
        ground_pose = ground_pose.unsqueeze(1)
        ground_pose = ground_pose.expand(-1, self.conf["num_blocks"], -1, -1)

        pred_vis = predictions[:, :, :, 2]
        ground_vis = poses[:, :, 2]
        ground_vis = ground_vis.unsqueeze(1)
        ground_vis = ground_vis.expand(-1, self.conf["num_blocks"], -1)

        binary_crossentropy = nn.BCELoss()

        vis_loss = binary_crossentropy(pred_vis, ground_vis)

        pose_loss = elastic_net_loss_paper(pred_pose, ground_pose)

        loss = vis_loss * 0.01 + pose_loss

        bboxes = train_objects["bbox"]

        distance_meassures = torch.FloatTensor(len(bboxes))

        for i in range(len(bboxes)):
            width = torch.abs(bboxes[i, 0] - bboxes[i, 2])
            height = torch.abs(bboxes[i, 1] - bboxes[i, 3])

            distance_meassures[i] = torch.max(width, height).item()

        self.pose_train_accuracies.append(torch.mean(torch.Tensor(eval_pck_batch(pred_pose[:, -1, :, 0:2], ground_pose[:, -1, :, 0:2], trans_matrices, distance_meassures))).item())


        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.train_writer.write([self.iteration, loss.item()])
        self.iteration = self.iteration + 1

        print("iteration {} loss {}".format(self.iteration, loss.item()))

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            pck_bb_02 = []
            pck_bb_01 = []
            pck_upper_02 = []

            self.create_dynamic_folders()

            mean_pose = torch.mean(torch.Tensor(self.pose_train_accuracies)).item()
            self.train_accuracy_writer.write([self.iteration, mean_pose])

            self.pose_train_accuracies = []

            for batch_idx, val_data in enumerate(self.val_loader):
                val_images = val_data["normalized_frames"].to(self.device)
                val_poses = val_data["normalized_poses"].to(self.device)

                trans_matrices = val_data["trans_matrices"].to(self.device)

                val_images = val_images.squeeze(dim=0)
                val_poses = val_poses.squeeze(dim=0)
                trans_matrices = trans_matrices.squeeze(dim=0)

                _, predictions, _, _ = self.model(val_images)
                predictions = predictions.squeeze(dim=0)

                if predictions.dim() == 2:
                    predictions = predictions.unsqueeze(0)

                # save predictions
                image = val_images[0].permute(1, 2, 0)
                gt_poses = val_poses

                # get distance meassures
                bboxes = val_data["bbox"].squeeze(dim=0)
                distance_meassures = torch.FloatTensor(len(bboxes))

                for i in range(len(bboxes)):
                    width = torch.abs(bboxes[i, 0] - bboxes[i, 2])
                    height = torch.abs(bboxes[i, 1] - bboxes[i, 3])

                    distance_meassures[i] = torch.max(width, height).item()

                pck_bb_02.extend(eval_pck_batch(predictions[:, :, 0:2], gt_poses[:, :, 0:2], trans_matrices, distance_meassures))
                pck_bb_01.extend(eval_pck_batch(predictions[:, :, 0:2], gt_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.1))
                #pck_upper_02.append(eval_pcku_batch(predictions[:, :, 0:2], gt_poses[:, :, 0:2], trans_matrices, compute_upperbody=True))

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

        torch.save(self.model.state_dict(), "experiments/{}/weights/weights_{:08d}".format(self.experiment_name, self.iteration))

        mean_bb_02 = torch.mean(torch.FloatTensor(pck_bb_02)).item()
        mean_bb_01 = torch.mean(torch.FloatTensor(pck_bb_01)).item()
        #mean_upper_02 = torch.mean(torch.FloatTensor(pck_upper_02)).item()
        mean_upper_02 = 0
        self.val_writer.write([self.iteration, mean_bb_02, mean_bb_01, mean_upper_02])
        return mean_bb_02

    def test(self, pretrained_model=None):
        with torch.no_grad():

            test_ds = PennActionDataset("/data/mjakobs/data/pennaction/", train=False, use_gt_bb=True)
            if pretrained_model is not None:
                self.preparation()
                self.model.load_state_dict(torch.load(pretrained_model, map_location=self.device))
    
            self.model.eval()
            
            pck_bb_02 = []
            pck_bb_01 = []

            for i in range(len(test_ds)):
                print(i)
                test_data = test_ds[i]
                start = 0
                nr_frames = len(test_data["normalized_frames"])
                stop = min(start + 20, nr_frames)
                need_stop = False
                while not need_stop:
                    test_images = test_data["normalized_frames"][start:stop].to(self.device)
                    test_poses = test_data["normalized_poses"][start:stop].to(self.device)
                    trans_matrices = test_data["trans_matrices"][start:stop].to(self.device)
                    original_window_sizes = test_data["original_window_size"][start:stop]

                    _, predictions, _, _ = self.model(test_images)
                    del test_images
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


                    predictions = predictions.squeeze(dim=0)

                    # get distance meassures
                    distance_meassures = torch.IntTensor(len(original_window_sizes))

                    for i in range(len(distance_meassures)):
                        distance = original_window_sizes[i][0]
                        distance_meassures[i] = original_window_sizes[i][0]

                    pck_bb_02.extend(eval_pck_batch(predictions[:, :, 0:2], test_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.2))
                    pck_bb_01.extend(eval_pck_batch(predictions[:, :, 0:2], test_poses[:, :, 0:2], trans_matrices, distance_meassures, threshold=0.1))

                    if stop == nr_frames:
                        need_stop = True
                    else:
                        start = stop
                        stop = min(stop + 20, nr_frames)

                    del predictions
                    del distance_meassures
                    del test_poses
                    del trans_matrices
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print("len", str(len(pck_bb_02)))
        mean_bb_02 = torch.mean(torch.FloatTensor(pck_bb_02)).item()
        mean_bb_01 = torch.mean(torch.FloatTensor(pck_bb_01)).item()
        return mean_bb_02, mean_bb_01

class MPIIExperiment(ExperimentBase):
    def __init__(self, conf, pretrained_model=None):
        super().__init__(conf)
        self.pretrained_model = pretrained_model

    def preparation(self):

        if "augmentation_amount" in self.conf:
            aug_amount = self.conf["augmentation_amount"]
        else:
            aug_amount = 1

        print(self.conf["use_saved_tensors"])
        self.ds_train = MPIIDataset("/data/mjakobs/data/mpii/", train=True, val=False, use_random_parameters=self.conf["use_random_parameters"], use_saved_tensors=self.conf["use_saved_tensors"], augmentation_amount=aug_amount)
        self.ds_val = MPIIDataset("/data/mjakobs/data/mpii/", train=True, val=True, use_random_parameters=False, use_saved_tensors=self.conf["use_saved_tensors"])

        if self.conf["num_blocks"] == 1:
            self.model = Mpii_1(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 2:
            self.model = Mpii_2(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 4:
            self.model = Mpii_4(num_context=self.conf["nr_context"]).to(self.device)
        if self.conf["num_blocks"] == 8:
            self.model = Mpii_8(num_context=self.conf["nr_context"]).to(self.device)

        if self.pretrained_model is not None:
            print("Using pretrained model at " + self.pretrained_model)
            self.model.load_state_dict(torch.load(self.pretrained_model, map_location=self.device))


        train_indices, val_indices = self.limit_dataset(include_test=False)

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

        if "lr_milestones" in self.conf:
            milestones = self.conf["lr_milestones"]
        else:
            milestones = [20000000] # basically, never use lr scheduler

        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.2)

        self.train_writer.write(["iteration", "loss"])
        self.val_writer.write(["iteration", "pckh_0.5", "pckh_0.2"])

    def train(self, train_objects):

        self.model.train()

        images = train_objects["normalized_image"].to(self.device)
        poses = train_objects["normalized_pose"].to(self.device)

        output, _, _, _ = self.model(images)

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
        self.lr_scheduler.step()
        self.iteration = self.iteration + 1

        print("iteration {} loss {}".format(self.iteration, loss.item()))


    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            val_accuracy_05 = []
            val_accuracy_02 = []
            self.create_dynamic_folders()

            for batch_idx, val_data in enumerate(self.val_loader):
                val_images = val_data["normalized_image"].to(self.device)

                _, predictions, _, _ = self.model(val_images)
                predictions = predictions.squeeze(dim=0)
                #predictions = predictions[-1, :, :, :].squeeze(dim=0)

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

class StemImageNet(ExperimentBase):

    def preparation(self):

        self.ds_train = ImageNet("/vol/corpora/vision/ILSVRC2012", split="train")
        self.ds_val = ImageNet("/vol/corpora/vision/ILSVRC2012", split="val")

        self.model = Stem()

        #train_indices, val_indices = self.limit_dataset(include_test=False)
        train_indices = list(range(len(ds_train)))
        val_indices = list(range(len(ds_val)))
        
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
        print(train_objects)
    def evaluate(self):
        print('not implemented')
