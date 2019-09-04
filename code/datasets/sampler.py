import torch
import torch.utils.data as data

import random

import os
import re
import glob

import scipy.io as sio
import numpy as np
import pandas as pd

from skimage import io
from skimage.transform import resize

from deephar.image_processing import center_crop, rotate_and_crop, normalize_channels
from deephar.utils import transform_2d_point, translate, scale, flip_h, superflatten, transform_pose, get_valid_joints

#based on https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
class ImbalancedDatasetSampler(data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx]["class"]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class EqualFrameLengthSampler(data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, use_random=True, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        if use_random:
            random.shuffle(self.indices)

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        self.frame_lengths = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in self.frame_lengths:
                self.frame_lengths[label].append(idx)
            else:
                self.frame_lengths[label] = [idx]

        for item in self.frame_lengths.keys():
            if len(self.frame_lengths[item]) % 2 == 1:
                self.frame_lengths[item] = self.frame_lengths[item][1:]


    def _get_label(self, dataset, idx):
        return dataset[idx]["sequence_length"].item()

    def __iter__(self):
        return self

    def __next__(self):
        keys = list(self.frame_lengths.keys())
        random.shuffle(keys)

        for key in keys:
            if len(self.frame_lengths[key]) == 0:
                continue

            sample_indices = random.sample(self.frame_lengths[key], 2)
            samples = []
            for sample_idx in sample_indices:
                samples.append(sample_idx)
                self.frame_lengths[key].remove(sample_idx)
            return samples

        raise StopIteration

    
    def __len__(self):
        return self.num_samples
