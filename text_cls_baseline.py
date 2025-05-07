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



def train_bert(model, classifier, train_loader, device="cuda"):

    model.train()
    classifier.train()

    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids, attention_mask, target = data["input_ids"].to(device), data["attention_mask"].to(device), target.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        feature = outputs.hidden_states[-1][:, 0, :]
        output = classifier(feature)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss
    

def eval_bert(model, classifier, test_loader, device="cuda"):
    model.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            input_ids, attention_mask, target = data["input_ids"].to(device), data["attention_mask"].to(device), target.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            feature = outputs.hidden_states[-1][:, 0, :]
            output = classifier(feature)

            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
        f' ({100. * correct / len(test_loader.dataset):.0f}%)')
    
    return correct / len(test_loader.dataset)


def main():

    parent_dir = f"saved_text_cls/baseline"
    os.makedirs(parent_dir, exist_ok=True)


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use CUDA or not: {DEVICE}")

    NUM_EXAMPLES = None

    DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
    # DATASET_NAMES = [args.dataset]


    METADATA_DATASET = dict()
    for dataset_name in DATASET_NAMES:

        METADATA_DATASET[dataset_name] = dict()
        dataloader = load_textclassification_data(dataset_name, maxsize=NUM_EXAMPLES, maxsize_for_each_class=100)

        METADATA_DATASET[dataset_name]["train_loader"] = dataloader[0]
        METADATA_DATASET[dataset_name]["test_loader"] = dataloader[2]

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


    def train(num_epochs=1, device=DEVICE):

        for dataset_name in DATASET_NAMES:

            print(f"Training {dataset_name}...")
            model = BertModel.from_pretrained('bert-base-uncased').to(device)
            classifier = nn.Linear(768, METADATA_DATASET[dataset_name]["num_classes"]).to(device)

            for epoch in range(num_epochs):
                avg_loss = train_bert(model=model, classifier=classifier, train_loader=METADATA_DATASET[dataset_name]["train_loader"], device=device)
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
                if epoch == 1:
                    acc2 = eval_bert(model=model, classifier=classifier, test_loader=METADATA_DATASET[dataset_name]["test_loader"], device=device)
                    METADATA_DATASET[dataset_name]["accuracy_2"] = acc2

            ft_extractor_path = f'{parent_dir}/{dataset_name}_bert.pth'
            torch.save(model.state_dict(), ft_extractor_path)
            classifier_path = f'{parent_dir}/{dataset_name}_classifier.pth'
            torch.save(classifier.state_dict(), classifier_path)

            acc = eval_bert(model=model, classifier=classifier, test_loader=METADATA_DATASET[dataset_name]["test_loader"], device=device)
            METADATA_DATASET[dataset_name]["accuracy"] = acc


    train(num_epochs=10, device=DEVICE)

    result_file = f"{parent_dir}/accuracy.txt"

    with open(result_file, 'a') as file:
        for dataset_name in DATASET_NAMES:
            file.write(f"Dataset: {dataset_name}, Accuracy_2: {METADATA_DATASET[dataset_name]['accuracy_2']}, Accuracy: {METADATA_DATASET[dataset_name]['accuracy']} \n")


if __name__ == '__main__':
    main()
