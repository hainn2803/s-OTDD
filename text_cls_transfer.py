import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_textclassification_data
from models.resnet import ResNet18, ResNet50
from otdd.pytorch.distance import DatasetDistance
from trainer import *
import os
import random
import json
from datetime import datetime, timedelta

from transformers import BertTokenizer, BertModel
import argparse

from text_cls import train_bert, eval_bert


def main():

    parser = argparse.ArgumentParser(description='OTDD')
    parser.add_argument('--dataset', default='AG_NEWS', help='dataset name')
    parser.add_argument('--num-epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')


    args = parser.parse_args()

    parent_dir = f"saved_text_cls"
    pretrained_weights = parent_dir + "/pretrained_weights" # save pretrained weights, already had
    adapt_weights = parent_dir + "/adapt_weights" # to save fine-tuned weights
    os.makedirs(adapt_weights, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use CUDA or not: {DEVICE}")

    SOURCE_DATASET = [args.dataset]
    TARGET_DATASET = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
    DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]

    METADATA_DATASET = dict()
    for dataset_name in DATASET_NAMES:

        METADATA_DATASET[dataset_name] = dict()

        if dataset_name == "AG_NEWS":
            METADATA_DATASET[dataset_name]["num_classes"] = 4

        elif dataset_name == "DBpedia":
            METADATA_DATASET[dataset_name]["num_classes"] = 14

        elif dataset_name == "YelpReviewPolarity":
            METADATA_DATASET[dataset_name]["num_classes"] = 2

        elif dataset_name == "YelpReviewFull":
            METADATA_DATASET[dataset_name]["num_classes"] = 5

        elif dataset_name == "YahooAnswers":
            METADATA_DATASET[dataset_name]["num_classes"] = 10

        elif dataset_name == "AmazonReviewPolarity":
            METADATA_DATASET[dataset_name]["num_classes"] = 2

        elif dataset_name == "AmazonReviewFull":
            METADATA_DATASET[dataset_name]["num_classes"] = 5
    
        dataloader = load_textclassification_data(dataset_name, maxsize=None, maxsize_for_each_class=100)

        METADATA_DATASET[dataset_name]["train_loader"] = dataloader[0]
        METADATA_DATASET[dataset_name]["test_loader"] = dataloader[2]

    
        METADATA_DATASET[dataset_name]["pretrained_weights"] = f"{pretrained_weights}/{dataset_name}_bert.pth"
    
    for i in range(len(SOURCE_DATASET)):
        for j in range(len(TARGET_DATASET)):
            
            source_dataset = SOURCE_DATASET[i]
            target_dataset = TARGET_DATASET[j]

            if source_dataset == target_dataset:
                continue
            
            else:

                extractor = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
                classifier = nn.Linear(768, METADATA_DATASET[target_dataset]["num_classes"]).to(DEVICE)

                additional_weights = torch.load(METADATA_DATASET[source_dataset]["pretrained_weights"], map_location=DEVICE)
                extractor.load_state_dict(additional_weights, strict=False)

                for epoch in range(args.num_epochs):
                    train_bert(model=extractor, classifier=classifier, train_loader=METADATA_DATASET[target_dataset]["train_loader"], device=DEVICE)
                    
                acc = eval_bert(model=extractor, classifier=classifier, test_loader=METADATA_DATASET[target_dataset]["test_loader"], device=DEVICE)

                ft_extractor_path = f'{adapt_weights}/{source_dataset}_{target_dataset}_bert.pth'
                torch.save(extractor.state_dict(), ft_extractor_path)

                classifier_path = f'{adapt_weights}/{source_dataset}_{target_dataset}_classifier.pth'
                torch.save(classifier.state_dict(), classifier_path)

                print(f"Source dataset: {source_dataset}, target dataset: {target_dataset}, accuracy: {acc}")

                result_file = f"{adapt_weights}/adapt_result.txt"

                with open(result_file, 'a') as file:
                    file.write(f"Source dataset: {source_dataset}, target dataset: {target_dataset}, accuracy: {acc} \n")

if __name__ == '__main__':
    main()

