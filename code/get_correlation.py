import torch
from scipy.stats import pearsonr
import time
from torch_geometric.loader import DataLoader
from prepare_dataset import LabelledDataset
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from models import GCNN, GAT, GraphSAGE
from distance_functions import *


torch.manual_seed(0)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def enable_dropout(model):
    """
    Enable dropout layers during test-time.

    Args:
    model (torch.nn.Module): The model to enable dropout layers for.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


DISTANCE_TORCH = {
    'square': square_distance,
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
    'manhattan': manhattan_distance,
    'hyperbolic': hyperbolic_distance
}

distances = ['square', 'euclidean', 'cosine', 'manhattan']
features_types = ['one_hot', 'physicochemical', 'blosum', 'bert', 'lstm']
features_dim = [20, 7, 25, 1024, 1024]
current_idx = 4
out_dim = 256


def get_mse(actual, predicted):
    """
    Compute the mean squared error between actual and predicted scores.

    Args:
    actual (numpy.ndarray): Actual scores.
    predicted (numpy.ndarray): Predicted scores.

    Returns:
    numpy.ndarray: The mean squared error between actual and predicted scores.
    """
    loss = ((actual - predicted) ** 2).mean(axis=0)
    return loss


def get_acc(actual, predicted, threshold=0.2):
    """
    Compute the accuracy based on a threshold for the absolute difference between actual and predicted scores.

    Args:
    actual (numpy.ndarray): Actual scores.
    predicted (numpy.ndarray): Predicted scores.
    threshold (float, optional): Threshold for the absolute difference. Defaults to 0.2.

    Returns:
    float: The accuracy based on the threshold.
    """
    count = 0
    for i in range(len(actual)):
        if abs(actual[i] - predicted[i]) < threshold:
            count += 1
    return count / len(actual)


for current_distance in distances:
    model_path = f"../models/GAT_{features_types[current_idx]}_{current_distance}_{out_dim}.pt"

    model = GAT(in_feat=features_dim[current_idx], out_feat=out_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    proteins_dir = os.path.join("../processed_graphs/", features_types[current_idx])
    csv_file = "../TM_scores.csv"
    dataset = LabelledDataset(csv_file, proteins_dir)
    size = dataset.n_samples
    split = 0.95
    trainset, testset = torch.utils.data.random_split(dataset, [math.floor(split * size), size - math.floor(split * size)])
    trainloader = DataLoader(dataset=trainset, batch_size=100, num_workers=5)
    testloader = DataLoader(dataset=testset, batch_size=100, num_workers=5)

    print("Test dataset size:", len(testloader.dataset))
    predictions = []
    labels = []
    start_time = time.time()
    with torch.no_grad():
        for (prot_1, prot_2, label) in testloader:
            prot_1 = prot_1.to(device)
            prot_2 = prot_2.to(device)
            embedding_1 = model(prot_1)
            embedding_2 = model(prot_2)

            distance = DISTANCE_TORCH[current_distance](embedding_1, embedding_2)
            label = label.numpy().flatten()
            distance = distance.cpu().numpy().flatten()
            labels = np.append(labels, label)
            predictions = np.append(predictions, distance)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"Correlation with {features_types[current_idx]} features, {current_distance} distance, and output dimension of {out_dim} is:")
    corr, _ = pearsonr(labels, predictions)
    print(f"Pearson's correlation: {corr:.3f}")
