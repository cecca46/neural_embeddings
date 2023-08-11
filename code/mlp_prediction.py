import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from collections import Counter
from torch import nn
from models import GCNN, GAT, MulticlassClassification
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from models import GCNN, GAT
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.model_selection import KFold
from prepare_dataset import *


device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")


def flatten(l):
    """
    Flatten a nested list.

    Args:
    l (list): The nested list to flatten.

    Returns:
    list: The flattened list.
    """
    return [item for sublist in l for item in sublist]


dinstances = ['square', 'euclidean', 'cosine', 'manhattan']
features_types = ['one_hot', 'physicochemical', 'blosum', 'bert', 'lstm']
features_dim = [20, 7, 25, 1024, 1024]
current_idx = 4
out_dim = 256
lines_to_write = []

for current_distance in dinstances:
    model_path = f"../models/GAT_{features_types[current_idx]}_{current_distance}_{out_dim}.pt"
    print(f"Using model from {model_path}")

    model = GAT(in_feat=features_dim[current_idx], out_feat=out_dim)
    model.load_state_dict(torch.load(model_path))

    proteins_dir = os.path.join("../processed_graphs/", features_types[current_idx])
    protein_files = [f for f in listdir(proteins_dir) if isfile(join(proteins_dir, f))]

    meta_info = pd.read_csv("../data/kinase_dataframe.csv")

    meta_info = meta_info.dropna(subset=['Group'])
    remove = ['Atypical', 'Other', 'RGC']
    meta_info = meta_info[~meta_info['Group'].isin(remove)]
    meta_info = meta_info.reset_index(drop=True)
    embeddings = []
    y_values = []
    with tqdm(total=meta_info.shape[0]) as pbar:
        for index, row in meta_info.iterrows():
            pbar.update(1)
            try:
                y = meta_info.iloc[index, 34]
                name = meta_info.iloc[index, 1]
                name = name + '.pt'
                protein = torch.load(os.path.join(proteins_dir, name))
                embedding = torch.squeeze(model(protein)).detach().numpy()
                embeddings.append(embedding)
                y_values.append(y)
            except:
                pass

    embeddings = np.array(embeddings)
    y_values = np.array(y_values)
    y_values = np.unique(y_values, return_inverse=True)[1]
    unique, counts = np.unique(y_values, return_counts=True)

    k_folds = 5
    num_epochs = 500
    loss_function = nn.CrossEntropyLoss()

    accuracies = []
    precisions = []
    recalls = []
    fscores = []

    dataset = SimpleDataset(embeddings, y_values)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    print("--------------------------------")

    for fold, (train_ids, test_ids) in enumerate(kfold.split(embeddings, y_values)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=50, sampler=train_subsampler
        )
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=50, sampler=test_subsampler
        )

        n_classes = len(np.unique(y_values))
        network = MulticlassClassification(out_dim, n_classes)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        for epoch in range(0, num_epochs):
            print(f"Starting epoch {epoch}")
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

        print("Training process has finished.")

        print("Starting testing")

        correct, total = 0, 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.append(predicted.numpy())
                all_labels.append(targets.numpy())
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        precision, recall, fscore, _ = precision_recall_fscore_support(flatten(all_labels), flatten(all_predictions), average='weighted', zero_division=0)
        accuracy = accuracy_score(flatten(all_labels), flatten(all_predictions))
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    print("Model is", features_types[current_idx] + "_" + current_distance + "_" + str(out_dim))
    print("Average accuracy is", np.mean(accuracies))
    print("Average precision is", np.mean(precisions))
    print("Average recall is", np.mean(recalls))
    print("Average fscore is", np.mean(fscores))
