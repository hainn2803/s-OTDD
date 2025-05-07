import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
from models.resnet import ResNet50
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.sotdd import compute_pairwise_distance
from trainer import *
import os
import random
from datetime import datetime, timedelta
import time
from torch.utils.data import Dataset, DataLoader
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")

OTDD_MAX_SIZE = 5000
SOTDD_MAX_SIZE = 50000


def main():
    parser = argparse.ArgumentParser(description='Arguments for Augmentation')
    parser.add_argument('--parent_dir', type=str, default="saved_augmentation", help='Parent directory')
    parser.add_argument('--seed', type=int, default=1, help='Seed')

    args = parser.parse_args()
        
    parent_dir = f"{args.parent_dir}/aug_{str(args.seed)}"
    os.makedirs(parent_dir, exist_ok=True)

    def create_data():
        brightness = random.uniform(0.1, 0.9)
        contrast = random.uniform(0.1, 0.9)
        saturation = random.uniform(0.1, 0.9)
        hue = random.uniform(0, 0.5)
        print(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue}")
        with open(result_file, 'a') as file:
            file.write(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue} \n")
        datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
        imagenet = load_imagenet(datadir=datadir_tiny_imagenet, 
                                resize=32, 
                                tiny=True, 
                                augmentations=True, 
                                brightness=brightness, 
                                contrast=contrast, 
                                saturation=saturation, 
                                maxsize=None,
                                hue=hue)
        imagenet_trainset = imagenet[1]["train"]
        imagenet_testset = imagenet[1]["test"]
        imagenet_trainloader = imagenet[0]["train"]
        imagenet_testloader = imagenet[0]["test"]
        
        datadir_cifar10 = "data/CIFAR10"
        cifar10 = load_torchvision_data("CIFAR10",
                                        valid_size=0, 
                                        download=False, 
                                        maxsize=None, 
                                        datadir=datadir_cifar10)
        cifar10_trainset = cifar10[1]["train"]
        cifar10_testset = cifar10[1]["test"]
        cifar10_trainloader = cifar10[0]["train"]
        cifar10_testloader = cifar10[0]["test"]

        def save_data(data_set, saved_tensor_path):
            list_images = list()
            list_labels = list()
            for img, label in data_set:
                list_images.append(img)
                list_labels.append(label)
            tensor_images = torch.stack(list_images)
            tensor_labels = torch.tensor(list_labels)
            torch.save((tensor_images, tensor_labels), saved_tensor_path)
            print(f"Number of data: {len(list_images)}, Save data into {saved_tensor_path}")

        save_data(data_set=imagenet_trainset, saved_tensor_path=f"{parent_dir}/transformed_train_imagenet.pt")
        save_data(data_set=imagenet_testset, saved_tensor_path=f"{parent_dir}/transformed_test_imagenet.pt")
        save_data(data_set=cifar10_trainset, saved_tensor_path=f"{parent_dir}/transformed_train_cifar10.pt")
        save_data(data_set=cifar10_testset, saved_tensor_path=f"{parent_dir}/transformed_test_cifar10.pt")

        return {
            "cifar10": {
                    "trainset": cifar10_trainset, 
                    "testset": cifar10_testset,
                    "trainloader": cifar10_trainloader, 
                    "testloader": cifar10_testloader
                    },
            "imagenet": {
                    "trainset": imagenet_trainset,
                    "testset": imagenet_testset,
                    "trainloader": imagenet_trainloader, 
                    "testloader": imagenet_testloader
                    }
        }


    # Train model, retrieve accuracy
    def transfer_learning(train_imagenet_loader, test_imagenet_loader, train_cifar10_loader, test_cifar10_loader, num_epochs_pretrain=300, num_epochs_adapt=30, device=DEVICE):
        print("Training backbone in ImageNet...")
        # Pretrain ImageNet model
        imagenet_feature_extractor = ResNet50().to(device)
        imagenet_classifier = nn.Linear(imagenet_feature_extractor.latent_dims, 200).to(device)
        feature_extractor_optimizer = optim.SGD(imagenet_feature_extractor.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        classifier_optimizer = optim.SGD(imagenet_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        for epoch in range(1, num_epochs_pretrain + 1):
            train_loss = train(feature_extractor=imagenet_feature_extractor,
                                classifier=imagenet_classifier,
                                device=device,
                                train_loader=train_imagenet_loader,
                                epoch=epoch,
                                criterion=nn.CrossEntropyLoss(),
                                ft_extractor_optimizer=feature_extractor_optimizer,
                                classifier_optimizer=classifier_optimizer)
            print(f"In training backbone in ImageNet, epoch {epoch}, training loss: {train_loss}")
        frozen_module(imagenet_feature_extractor)

        ft_extractor_path = f'{parent_dir}/imagenet_ft_extractor.pth'
        torch.save(imagenet_feature_extractor.state_dict(), ft_extractor_path)

        classifier_path = f'{parent_dir}/imagenet_classifier.pth'
        torch.save(imagenet_classifier.state_dict(), classifier_path)

        imagenet_acc_no_adapt = test_func(feature_extractor=imagenet_feature_extractor, classifier=imagenet_classifier, device=device, test_loader=test_imagenet_loader)

        print("Training transfer learning in CIFAR10...")
        # Transfer learning on CIFAR10
        cifar10_classifier = nn.Linear(imagenet_feature_extractor.latent_dims, 10).to(device)
        cifar10_classifier_optimizer = optim.SGD(cifar10_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        for epoch in range(1, num_epochs_adapt + 1):
            train_loss = train(feature_extractor=imagenet_feature_extractor,
                                classifier=cifar10_classifier,
                                device=device,
                                train_loader=train_cifar10_loader,
                                epoch=epoch,
                                criterion=nn.CrossEntropyLoss(),
                                ft_extractor_optimizer=None,
                                classifier_optimizer=cifar10_classifier_optimizer)
            print(f"In training transfer learning in CIFAR10, epoch {epoch}, training loss: {train_loss}")

        classifier_path = f'{parent_dir}/cifar10_classifier.pth'
        torch.save(cifar10_classifier.state_dict(), classifier_path)

        cifar10_acc_adapt = test_func(feature_extractor=imagenet_feature_extractor, classifier=cifar10_classifier, device=device, test_loader=test_cifar10_loader)
        print(f"Accuracy of CIFAR10: {cifar10_acc_adapt}")
        return cifar10_acc_adapt


    class CustomTensorDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]


    def get_dataloader(datadir, maxsize=None, batch_size=64):
        images_tensor, labels_tensor = torch.load(datadir)
        if maxsize is not None:
            if maxsize < images_tensor.size(0):
                indices = torch.randperm(images_tensor.size(0))[:maxsize]
                selected_images = images_tensor[indices]
                selected_labels = labels_tensor[indices]
                dataset = CustomTensorDataset(selected_images, selected_labels)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            else:
                dataset = CustomTensorDataset(images_tensor, labels_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        else:
            dataset = CustomTensorDataset(images_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader


    print("Start creating data")
    DATA_DICT = create_data()
    print("Finish creating data")

    imagenet_trainloader = DATA_DICT["imagenet"]["trainloader"]
    cifar10_trainloader = DATA_DICT["cifar10"]["trainloader"]
    imagenet_testloader = DATA_DICT["imagenet"]["testloader"]
    cifar10_testloader = DATA_DICT["cifar10"]["testloader"]

    cifar10_acc_adapt = transfer_learning(train_imagenet_loader=imagenet_trainloader, 
                                            test_imagenet_loader=imagenet_testloader, 
                                            train_cifar10_loader=cifar10_trainloader, 
                                            test_cifar10_loader=cifar10_testloader,
                                            num_epochs_pretrain=300, 
                                            num_epochs_adapt=30,
                                            device=DEVICE)

    cifar10_dataloader = get_dataloader(datadir=f'{parent_dir}/transformed_train_cifar10.pt', maxsize=SOTDD_MAX_SIZE, batch_size=64)
    imagenet_dataloader = get_dataloader(datadir=f'{parent_dir}/transformed_train_imagenet.pt', maxsize=SOTDD_MAX_SIZE, batch_size=64)
    
    # Compute s-OTDD
    num_projection = 100000
    kwargs = {
        "dimension": 32,
        "num_channels": 3,
        "num_moments": 5,
        "use_conv": True,
        "precision": "float",
        "p": 2,
        "chunk": 1000
    }
    list_dataset = [cifar10_dataloader, imagenet_dataloader]
    sotdd_dist = compute_pairwise_distance(list_D=list_dataset, device="cpu", num_projections=num_projection, **kwargs)[0].item()
    print(sotdd_dist)

    cifar10_dataloader = get_dataloader(datadir=f'{parent_dir}/transformed_train_cifar10.pt', maxsize=OTDD_MAX_SIZE, batch_size=64)
    imagenet_dataloader = get_dataloader(datadir=f'{parent_dir}/transformed_train_imagenet.pt', maxsize=OTDD_MAX_SIZE, batch_size=64)

    # Compute OTDD (Exact)
    otdd_dist = DatasetDistance(cifar10_dataloader,
                                imagenet_dataloader,
                                inner_ot_method='exact',
                                debiased_loss=True,
                                p=2,
                                entreg=1e-3,
                                device="cpu")
    otdd_d = otdd_dist.distance(maxsamples=OTDD_MAX_SIZE).item()

    with open(f'{parent_dir}/result.txt', 'a') as file:
        file.write(f"seed id: {seed_id}, accuracy: {cifar10_acc_adapt}, sotdd distance: {sotdd_dist}, otdd exact distance: {otdd_d}")


if __name__ == "__main__":
    main()
