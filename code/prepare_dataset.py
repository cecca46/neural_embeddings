import os
import torch
import glob
import numpy as np 
import random
import math
import pandas as pd
from os import listdir
from os.path import isfile
from torch.utils.data import Dataset 
from torch_geometric.loader import DataLoader 


class LabelledDataset(Dataset):
    def __init__(self, csv_file, processed_path):
        """
        Dataset class for labeled protein pairs.

        Args:
        csv_file (str): Path to the CSV file containing protein pairs and labels.
        processed_path (str): Path to the directory containing processed protein data.
        """
        self.csv_file = csv_file
        self.processed_path = processed_path
        self.reference_pairs = pd.read_csv(csv_file)
        self.n_samples = self.reference_pairs.shape[0]
        
    def __getitem__(self, index):
        """
        Get a specific item (protein pair and label) from the dataset.

        Args:
        index (int): Index of the item.

        Returns:
        tuple: Tuple containing the two proteins (torch.Tensor) and the label (torch.Tensor).
        """
        file_name1 = self.reference_pairs.iloc[index, 0].split('/')[3].split('.')[0]
        file_name2 = self.reference_pairs.iloc[index, 1].split('/')[3].split('.')[0]
        label = 1.0 - float(self.reference_pairs.iloc[index, 2])
        protein1 = torch.load(os.path.join(self.processed_path, file_name1 + '.pt'))
        protein2 = torch.load(os.path.join(self.processed_path, file_name2 + '.pt'))
        return protein1, protein2, torch.tensor(label)

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
        int: Number of samples in the dataset.
        """
        return self.n_samples


class SimpleDataset(Dataset):
    def __init__(self, embeddings, labels):
        """
        Dataset class for simple embeddings and labels.

        Args:
        embeddings (np.ndarray): Array of embeddings.
        labels (np.ndarray): Array of corresponding labels.
        """
        self.embeddings = embeddings
        self.labels = labels
        self.n_samples = embeddings.shape[0]

    def __getitem__(self, index):
        """
        Get a specific item (embedding and label) from the dataset.

        Args:
        index (int): Index of the item.

        Returns:
        tuple: Tuple containing the embedding (torch.Tensor) and the label (torch.Tensor).
        """
        return self.embeddings[index], self.labels[index]

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
        int: Number of samples in the dataset.
        """
        return self.n_samples
