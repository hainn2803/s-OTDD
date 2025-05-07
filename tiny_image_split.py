import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
from otdd.pytorch.method5 import compute_pairwise_distance
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data.sampler import SubsetRandomSampler
import time
from trainer import train, test_func, frozen_module
from models.resnet import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from PIL import Image
import random
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet

import torch.multiprocessing as mp



class SortedRelabeledSubset(Dataset):
    """
    A subset of a dataset at the specified indices, with labels remapped by 
    sorting unique labels so they become 0..(num_unique_labels-1).
    """
    def __init__(self, dataset, indices):
        """
        :param dataset: The original dataset (e.g., Tiny-ImageNet).
        :param indices: List/Tensor of indices that define this subset.
        """
        super().__init__()
        self.dataset = dataset

        # Ensure indices is a Python list (if it's a tensor).
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        self.indices = indices

        # Extract the relevant labels from the dataset
        original_targets = torch.tensor(dataset.targets)
        subset_targets = original_targets[self.indices]

        # Collect unique labels, sort them
        unique_labels = torch.unique(subset_targets).tolist()
        unique_labels.sort()

        # Build a map: old_label -> new_label
        self.label_map = {
            old_label: new_label 
            for new_label, old_label in enumerate(unique_labels)
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx]
        x, old_label = self.dataset[dataset_idx]
        new_label = self.label_map[old_label]  # map old -> new
        return x, new_label


def save_dataset(dataloader, filepath):
    """
    Iterates through a PyTorch DataLoader to collect all (x, y) 
    into two large tensors (x_all, y_all), then saves them to `filepath`.

    Note:
      - This requires loading the entire dataset into memory at once, 
        which may be infeasible for very large datasets.
    """
    x_all = []
    y_all = []

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    for x, y in dataloader:
        # x: (batch_size, C, H, W) or something similar
        # y: (batch_size,)
        x_all.append(x)
        y_all.append(y)
    print(len(x_all), len(y_all))
    # Concatenate along the first dimension (batch dimension)
    x_all = torch.cat(x_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    # Save as a tuple or dict
    torch.save((x_all, y_all), filepath)
    print(f"Saved data to {filepath}! x_all.shape={x_all.shape}, y_all.shape={y_all.shape}")


# (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

def random_split_into_tasks(
    trainset, 
    testset, 
    num_tasks=10, 
    labels_per_task=20, 
    batch_size=64, 
    num_workers=2
):
    """
    Randomly splits the dataset classes into `num_tasks` disjoint groups, each
    containing `labels_per_task` classes. Then builds SortedRelabeledSubsets 
    and DataLoaders so that each task's labels are re-labeled to [0..(K-1)].
    
    Returns:
        train_loaders, test_loaders: Two lists of length `num_tasks`. 
                                     Each element is a DataLoader for that task.
    """
    
    train_loaders = []
    test_loaders  = []

    all_labels_train = torch.tensor(trainset.targets)
    unique_labels = torch.unique(all_labels_train).tolist()
    random.shuffle(unique_labels)
    label_groups = []
    total_labels = len(unique_labels)
    assert total_labels == num_tasks * labels_per_task, (
        f"Expected exactly {num_tasks*labels_per_task} unique labels, but found {total_labels}."
    )

    for i in range(num_tasks):
        start_idx = i * labels_per_task
        end_idx   = start_idx + labels_per_task
        group     = unique_labels[start_idx:end_idx]
        label_groups.append(group)

    train_targets = torch.tensor(trainset.targets)
    test_targets  = torch.tensor(testset.targets)
    print(test_targets)
    print(len(torch.unique(train_targets)), len(torch.unique(test_targets)))

    indices_train = np.arange(len(trainset))
    indices_test = np.arange(len(testset))
    for group_idx, label_group in enumerate(label_groups):

        train_indices = list()
        test_indices = list()
        for label_id in label_group:
            chosen_indices_train = indices_train[trainset.targets == label_id]
            chosen_indices_test = indices_test[testset.targets == label_id]
            train_indices.extend(chosen_indices_train)
            test_indices.extend(chosen_indices_test)
        print(len(train_indices), len(test_indices))

        train_subset = SortedRelabeledSubset(trainset, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        train_loaders.append(train_loader)

        test_subset = SortedRelabeledSubset(testset, test_indices)
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"
    print(f"Use CUDA or not: {DEVICE}")

    datadir_tiny_imagenet = "data/tiny-imagenet-200"
    imagenet = load_imagenet(datadir=datadir_tiny_imagenet)

    imagenet_trainset = imagenet[1]["train"]
    imagenet_testset = imagenet[1]["test"]

    imagenet_trainloader = imagenet[0]["train"]
    imagenet_testloader = imagenet[0]["test"]

    num_tasks        = 10
    labels_per_task  = 200 // num_tasks
    batch_size       = 16
    
    train_loaders, test_loaders = random_split_into_tasks(
        trainset        = imagenet_trainset, 
        testset         = imagenet_testset,
        num_tasks       = num_tasks, 
        labels_per_task = labels_per_task,
        batch_size      = batch_size,
        num_workers     = 0
    )

    for task_id in range(num_tasks):
        save_dataset(dataloader=train_loaders[task_id], filepath=f"saved_split_task/data/trainset_{task_id}.pt")
        save_dataset(dataloader=test_loaders[task_id], filepath=f"saved_split_task/data/test_{task_id}.pt")

if __name__ == "__main__":
    main()