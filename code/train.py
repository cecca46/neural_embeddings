from models import * 
import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib
import math
import sklearn
import torch_optimizer as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.nn as nn
from torch_geometric.loader import DataLoader as DataLoader
from prepare_dataset import * 
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from distance_functions import *

torch.manual_seed(1)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

DISTANCE_TORCH = {
    'square': square_distance,
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
    'manhattan': manhattan_distance,
    'hyperbolic': hyperbolic_distance
}

def get_mse(actual, predicted):
    loss = ((actual - predicted) ** 2).mean(axis=0)
    return loss

def train(model, device, trainloader, optimizer, epoch):

    model.train()
    loss_func = nn.MSELoss()
    scheduler = MultiStepLR(optimizer, milestones=[1,5], gamma=0.5)
    for count, (protein_1, protein_2, label) in enumerate(trainloader):
        protein_1 = protein_1.to(device)
        protein_2 = protein_2.to(device)
        label = label.to(device)
        optimizer.zero_grad()
		#embedding for protein 1 and 2
        embedding_1 = model(protein_1)
        embedding_2 = model(protein_2)
        
        distance = DISTANCE_TORCH[current_distance](embedding_1, embedding_2)
        loss = loss_func(label, distance)
        loss.backward()
        optimizer.step()
	
    scheduler.step()
    print('Epoch %d ---- train loss %f ' %(epoch, loss))

def predict(model, device, loader):
	
    model.eval()
    predictions = []
    labels = []
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
    return labels, predictions

processed_path = '../processed_graphs/one_hot'
csv_file = '../TM_scores.csv'
dataset = LabelledDataset(csv_file, processed_path)
print ('Dataset size is %d' %(dataset.n_samples))
size = dataset.n_samples

current_distance = 'square'
out_dim = 256
in_dim = 7
print ('Using %s as distance function and output size of %d' %(current_distance, out_dim))
print (processed_path)

trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size)])
trainloader = DataLoader(dataset = trainset, batch_size = 100, num_workers = 5)
testloader = DataLoader(dataset = testset, batch_size = 100, num_workers = 5)

model = GCNN(in_feat = in_dim, out_feat = out_dim)
model.to(device)
num_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
min_loss = 100
n_epochs_stop = 10
for epoch in range(num_epochs):
    
    train(model, device, trainloader, optimizer, epoch)
    G, P = predict(model, device, testloader)
    loss = get_mse(G,P)
    print('Epoch %d ---- test loss %f ' %(epoch, loss))
    if (loss < min_loss):
        min_loss = loss
        min_loss_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), '../models/GCN_one_hot_' + str(current_distance) + '_' + str(out_dim) + '.pt')
    elif (loss > min_loss):
        epochs_no_improve += 1
    if (epoch > 5 and epochs_no_improve == n_epochs_stop):
        print('Early stopping!')
        break

print ('Min test MSE loss is %f at epoch %d' %(min_loss, min_loss_epoch)) 
print('Model saved')




