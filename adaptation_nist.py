import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.sotdd import compute_pairwise_distance
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.moments import compute_label_stats
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import json
import pickle
from wte.distance import WTE
from scipy.spatial import distance
from geoopt import Lorentz as Lorentz_geoopt
from hswfs_otdd.utils.hmds import HyperMDS
from hswfs_otdd.utils.bures_wasserstein import LabelsBW
from hswfs_otdd.hswfs.manifold.euclidean import Euclidean
from hswfs_otdd.hswfs.manifold.product import ProductManifold
from hswfs_otdd.hswfs.manifold.poincare import Poincare
from hswfs_otdd.hswfs.manifold.lorentz import Lorentz
from hswfs_otdd.hswfs.sw import sliced_wasserstein





LIST_DATASETS = ["MNIST", "FashionMNIST", "EMNIST", "KMNIST", "USPS"]
ACC_ADAPT = dict()
DIST = dict()

parent_dir = f"saved_nist/dist"
pretrained_path = parent_dir + "/pretrained_weights"
adapt_path = parent_dir + "/finetune_weights"

os.makedirs(parent_dir, exist_ok=True)
os.makedirs(pretrained_path, exist_ok=True)
os.makedirs(adapt_path, exist_ok=True)

DEVICE = "cuda"

# Load data
MAXSIZE_DIST = None
MAXSIZE_TRAINING = None

def generate_reference(num, dim_low, dim, attached_dim, seed=0):
    torch.manual_seed(seed)
    med = torch.rand(num, dim_low, dim_low).unsqueeze(0)
    s = dim/dim_low
    m = nn.Upsample(scale_factor=s, mode='bilinear')
    attached = torch.randn(num, attached_dim)
    return torch.cat((m(med).reshape(num, -1), attached), dim=1).float()


def create_dataset(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None):

    METADATA_DATASET = dict()
    for dataset_name in LIST_DATASETS:

        METADATA_DATASET[dataset_name] = dict()

        if dataset_name == "USPS":
            data_folders = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, datadir="data/USPS", maxsize_for_each_class=maxsize_for_each_class)
        else:
            data_folders = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, maxsize_for_each_class=maxsize_for_each_class)

        METADATA_DATASET[dataset_name]["train_loader"] = data_folders[0]['train']
        METADATA_DATASET[dataset_name]["train_set"] = data_folders[1]['train']
        METADATA_DATASET[dataset_name]["pretrained_extractor_path"] = f'{pretrained_path}/{dataset_name}/extractor_layers.pth'
        METADATA_DATASET[dataset_name]["pretrained_classifier_path"] = f'{pretrained_path}/{dataset_name}/fc_layers.pth'

        if dataset_name == "MNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "KMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "EMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 26
        elif dataset_name == "FashionMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "USPS":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        else:
            raise("Unknown src dataset")
    
    return METADATA_DATASET


def compute_otdd_distance(maxsamples=MAXSIZE_DIST, METADATA_DATASET=None):

    if METADATA_DATASET is None:
        METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0

    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]

            dist = DatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                    METADATA_DATASET[target_dataset]["train_loader"],
                                    inner_ot_method='exact',
                                    debiased_loss=True,
                                    p=2,
                                    entreg=1e-3,
                                    device="cpu")

            d = dist.distance(maxsamples = maxsamples).item()

            all_dist_dict[source_dataset][target_dataset] = d
            all_dist_dict[target_dataset][source_dataset] = d

            print(f'DIST({source_dataset}, {target_dataset})={d:8.2f}')

    return all_dist_dict


def compute_otdd_gaussian_distance(maxsamples=MAXSIZE_DIST, METADATA_DATASET=None):

    if METADATA_DATASET is None:
        METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0

    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]

            dist = DatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                    METADATA_DATASET[target_dataset]["train_loader"],
                                    inner_ot_method='gaussian_approx',
                                    sqrt_method='approximate',
                                    nworkers_stats=0,
                                    sqrt_niters=10,
                                    debiased_loss=True,
                                    p=2,
                                    entreg=1e-3,
                                    device="cpu")

            d = dist.distance(maxsamples = maxsamples).item()

            all_dist_dict[source_dataset][target_dataset] = d
            all_dist_dict[target_dataset][source_dataset] = d

            print(f'DIST({source_dataset}, {target_dataset})={d:8.2f}')

    return all_dist_dict


def compute_wte_distance(maxsamples=MAXSIZE_DIST, METADATA_DATASET=None):

    if METADATA_DATASET is None:
        METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0
    

    subdatasets = list()

    for i in range(len(LIST_DATASETS)):
        subdatasets.append(METADATA_DATASET[LIST_DATASETS[i]]["train_set"])

    reference = generate_reference(MAXSIZE_DIST, 4, 28, 10)
    wtes = WTE(subdatasets, label_dim=10, device=DEVICE, ref=reference.cpu(), maxsamples=MAXSIZE_DIST)
    wtes = wtes.reshape(wtes.shape[0], -1)
    wte_distance = distance.cdist(wtes, wtes, 'euclidean')

    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]
            all_dist_dict[source_dataset][target_dataset] = wte_distance[i][j].item()
            all_dist_dict[target_dataset][source_dataset] = wte_distance[i][j].item()
    return all_dist_dict


def compute_hswfs_distance(maxsamples=MAXSIZE_DIST, METADATA_DATASET=None, num_proj=500):

    if METADATA_DATASET is None:
        METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0
    
    subdatasets = list()

    for i in range(len(LIST_DATASETS)):
        subdatasets.append(METADATA_DATASET[LIST_DATASETS[i]]["train_set"])

    n_projs = num_proj
    scaling = 0.1
    d = 10
    n_epochs = 5000

    emb = LabelsBW(device="cpu", maxsamples=MAXSIZE_DIST)
    distance_array = emb.dissimilarity_for_all(subdatasets)
    lorentz_geoopt = Lorentz_geoopt()
    embedding = HyperMDS(d, lorentz_geoopt, torch.optim.Adam, scaling=scaling, loss="ads")
    mds, L = embedding.fit_transform(torch.tensor(distance_array, dtype=torch.float64), n_epochs=n_epochs, lr=1e-3)

    dist_mds = lorentz_geoopt.dist(mds[None], mds[:,None]).detach().cpu().numpy()
    diff_dist = np.abs(scaling * distance_array - dist_mds)
    data_X = [] # data
    data_Y = [] # labels
    for cac_idx, cac_dataset in enumerate(subdatasets):
        X, Y = emb.preprocess_dataset(cac_dataset)
        label_emb = mds[emb.class_num*cac_idx:emb.class_num*(cac_idx+1)].detach().numpy()
        labels = torch.stack([torch.from_numpy(label_emb[target])
                            for target in Y], dim=0).squeeze(1).to("cpu")
        data_X.append(X)
        data_Y.append(labels)
    d_y = data_Y[0].shape[1]
    manifolds = [Euclidean(28*28, device="cpu"), Lorentz(d_y, projection="horospheric", device="cpu")]
    product_manifold = ProductManifold(manifolds, torch.ones((2,), device="cpu")/np.sqrt(2))
    d_sw = np.zeros((len(subdatasets), len(subdatasets)))

    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]
            sw = sliced_wasserstein([data_X[i], data_Y[i]], [data_X[j], data_Y[j]], n_projs, product_manifold).item()
            all_dist_dict[source_dataset][target_dataset] = sw
            all_dist_dict[target_dataset][source_dataset] = sw

    return all_dist_dict


def compute_sotdd_distance(maxsamples=MAXSIZE_DIST, num_moments=1, num_projection=10000, METADATA_DATASET=None):

    if METADATA_DATASET is None:
        METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    list_dataset = list()
    for i in range(len(LIST_DATASETS)):
        dt_name = LIST_DATASETS[i]
        list_dataset.append(METADATA_DATASET[dt_name]["train_loader"])
    
    kwargs = {
        "dimension": 28 * 28,
        "num_channels": 1,
        "num_moments": num_moments,
        "use_conv": False,
        "precision": "float",
        "p": 2,
        "chunk": 1000
    }

    sw_list = compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=num_projection, **kwargs)

    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0

    k = 0
    for i in range(len(LIST_DATASETS)):
        for j in range(i + 1, len(LIST_DATASETS)):

            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]
            all_dist_dict[source_dataset][target_dataset] = sw_list[k].item()
            all_dist_dict[target_dataset][source_dataset] = sw_list[k].item()

            k += 1
    
    assert k == len(sw_list), "k != len(sw_list)"

    return all_dist_dict


def train_source(num_epoch_source=20, maxsamples=MAXSIZE_TRAINING, device=DEVICE):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    # TRAIN SOURCE PHASE
    ACC_NO_ADAPT = dict()
    for source in LIST_DATASETS:
        os.makedirs(f"{pretrained_path}/{source}", exist_ok=True)

        ft_extractor = FeatureExtractor(input_size=28).to(device)
        classifier = FullyConnectedNetwork(feat_dim=ft_extractor.feat_dim, num_classes=METADATA_DATASET[source]["num_classes"]).to(device)
        ft_extractor_optimizer = optim.Adam(ft_extractor.parameters(), lr=1e-3, weight_decay=1e-6)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

        list_acc = list()
        for epoch in range(1, num_epoch_source + 1):
            train(feature_extractor=ft_extractor, 
                classifier=classifier, 
                device=device,
                train_loader=METADATA_DATASET[source]["train_loader"], 
                criterion=nn.CrossEntropyLoss(), 
                ft_extractor_optimizer=ft_extractor_optimizer, 
                classifier_optimizer=classifier_optimizer)
        
            acc_no_adapt = test_func(feature_extractor=ft_extractor, classifier=classifier, device=device, test_loader=METADATA_DATASET[source]["test_loader"])
            list_acc.append(acc_no_adapt)

        ACC_NO_ADAPT[source] = list_acc
        
        ft_extractor_path = f'{pretrained_path}/{source}/extractor_layers.pth'
        torch.save(ft_extractor.state_dict(), ft_extractor_path)

        classifier_path = f'{pretrained_path}/{source}/fc_layers.pth'
        torch.save(classifier.state_dict(), classifier_path)

        with open(f"{pretrained_path}/{source}/accuracy.txt", "a") as file:
            for i in range(len(ACC_NO_ADAPT[source])):
                file.write(f"Epoch: {i}, accuracy: {ACC_NO_ADAPT[source][i]} \n")


def training_and_adaptation(num_epochs=10, maxsamples=MAXSIZE_TRAINING, device=DEVICE):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    # TRAIN ADAPTATION PHASE
    for i in range(len(LIST_DATASETS)):
        
        target = LIST_DATASETS[i]

        if target not in ACC_ADAPT:
            ACC_ADAPT[target] = dict()

        for j in range(len(LIST_DATASETS)):

            source = LIST_DATASETS[j]

            if target == source:
                continue
            
            else:
                
                os.makedirs(f"{adapt_path}/{target}/{source}", exist_ok=True)

                ft_extractor = FeatureExtractor(input_size=28).to(device)
                ft_extractor.load_state_dict(torch.load(METADATA_DATASET[source]["pretrained_extractor_path"]))
                ft_extractor_optimizer = None

                classifier = FullyConnectedNetwork(feat_dim=ft_extractor.feat_dim, num_classes=METADATA_DATASET[source]["num_classes"]).to(device)
                classifier.load_state_dict(torch.load(METADATA_DATASET[source]["pretrained_classifier_path"]))
                classifier.change_head(new_num_classes=METADATA_DATASET[target]["num_classes"])
                classifier = classifier.to(device)
                classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

                list_accuracy = list()
                for epoch in range(1, num_epochs + 1):
                    train(feature_extractor=ft_extractor, 
                        classifier=classifier, 
                        device=device, 
                        train_loader=METADATA_DATASET[target]["train_loader"], 
                        epoch=epoch,
                        criterion=nn.CrossEntropyLoss(), 
                        ft_extractor_optimizer=ft_extractor_optimizer, 
                        classifier_optimizer=classifier_optimizer)
                    
                    acc_adapt = test_func(ft_extractor, classifier, device, METADATA_DATASET[target]["test_loader"])
                    list_accuracy.append(acc_adapt)

                ACC_ADAPT[target][source] = list_accuracy

                ft_extractor_path = f'{adapt_path}/{target}/{source}/extractor_layers.pth'
                torch.save(ft_extractor.state_dict(), ft_extractor_path)

                classifier_path = f'{adapt_path}/{target}/{source}/fc_layers.pth'
                torch.save(classifier.state_dict(), classifier_path)

                with open(f"{adapt_path}/{target}/{source}/accuracy.txt", "a") as file:
                    for i in range(len(ACC_ADAPT[target][source])):
                        file.write(f"Epoch: {i}, accuracy: {ACC_ADAPT[target][source][i]} \n")

                print(f"In adaptation, source dataset {source}, target dataset: {target}, accuracy: {acc_adapt}")


if __name__ == "__main__":

    METADATA_DATASET = create_dataset(maxsamples=MAXSIZE_DIST)
    

    # DIST_otdd = compute_otdd_gaussian_distance(METADATA_DATASET=METADATA_DATASET)
    # dist_file_path = f'{parent_dir}/otdd_gaussian_distance.json'
    # with open(dist_file_path, 'w') as json_file:
    #     json.dump(DIST_otdd, json_file, indent=4)


    # DIST_otdd = compute_otdd_distance(METADATA_DATASET=METADATA_DATASET)
    # dist_file_path = f'{parent_dir}/otdd_exact_distance.json'
    # with open(dist_file_path, 'w') as json_file:
    #     json.dump(DIST_otdd, json_file, indent=4)

    num_moments = 7
    DIST_sotdd = compute_sotdd_distance(num_projection=10000, num_moments=num_moments, METADATA_DATASET=METADATA_DATASET)
    dist_file_path = f'{parent_dir}/sotdd_distance_num_moments_{num_moments}.json'
    with open(dist_file_path, 'w') as json_file:
        json.dump(DIST_sotdd, json_file, indent=4)

    # CUDA_VISIBLE_DEVICES=3 python3 adaptation_nist.py

    # DIST_sotdd = compute_wte_distance(METADATA_DATASET=METADATA_DATASET)
    # dist_file_path = f'{parent_dir}/wte_distance.json'
    # with open(dist_file_path, 'w') as json_file:
    #     json.dump(DIST_sotdd, json_file, indent=4)


    # DIST_sotdd = compute_hswfs_distance(num_proj=10000, METADATA_DATASET=METADATA_DATASET)
    # dist_file_path = f'{parent_dir}/hswfs_distance.json'
    # with open(dist_file_path, 'w') as json_file:
    #     json.dump(DIST_sotdd, json_file, indent=4)

    # train_source(num_epoch_source=20, maxsamples=MAXSIZE_TRAINING, device=DEVICE)
    # training_and_adaptation(num_epochs=10, maxsamples=MAXSIZE_TRAINING, device=DEVICE)

