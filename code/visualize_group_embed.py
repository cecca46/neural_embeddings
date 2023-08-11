import torch
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
import plotly.express as px

device = torch.device("cuda:3") if torch.cuda.is_available() else torch.cuda("cpu")

dinstances = ['square', 'euclidean', 'cosine', 'manhattan']#, 'hyperbolic']
features_types = ['one_hot', 'physicochemical', 'blosum', 'bert', 'lstm']
features_dim = [20, 7, 25, 1024, 1024]
#index of feats to use 0: one hot, 1: physicochemical, 2: blosum fts, 3: bert fts, 4: lstm
current_idx = 4
out_dim = 256

for current_distance in dinstances:
	model_path = '../models/GCNN_'+features_types[current_idx] + '_' + current_distance + '_' + str(out_dim) + '.pt'
	print ('Using model from %s' %model_path)

	model = GCNN(in_feat = features_dim[current_idx], out_feat = out_dim)
	model.load_state_dict(torch.load(model_path))

	proteins_dir = os.path.join('../processed_graphs/', features_types[current_idx])
	protein_files = [f for f in listdir(proteins_dir) if isfile(join(proteins_dir, f))]

	meta_info =  pd.read_csv('../data/kinase_dataframe.csv')
	meta_info = meta_info.dropna(subset=['Group'])
	remove = ['Atypical', 'Other', 'RGC']
	meta_info = meta_info[~meta_info['Group'].isin(remove)]
	meta_info = meta_info.reset_index(drop=True)
	embeddings = []
	fam_values = []

	with tqdm(total=meta_info.shape[0]) as pbar: 
		for index, row in meta_info.iterrows():
			pbar.update(1)
			try:
				fam = meta_info.iloc[index, 34]
				name = meta_info.iloc[index, 1]
				name = name + '.pt'
				protein = torch.load(os.path.join(proteins_dir, name))	
				embedding = torch.squeeze(model(protein)).detach().numpy()
				embeddings.append(embedding)	
				fam_values.append(fam)
			except: pass

	embeddings = np.array(embeddings)
	fam_values = np.array(fam_values)
	low_dim_embed = TSNE(n_components = 2).fit_transform(embeddings)
	pca = PCA(n_components = 2)
	pca_result = pca.fit_transform(embeddings)
	umap = UMAP(n_components=2, init='random')
	umap_result = umap.fit_transform(embeddings)


	df = pd.DataFrame()
	df["comp-1"] = low_dim_embed[:,0]
	df["comp-2"] = low_dim_embed[:,1]
	df["Fam"] = fam_values
	df["pca-one"] = pca_result[:,0]
	df["pca-two"] = pca_result[:,1]
	df["umap-1"] = umap_result[:, 0]
	df["umap-2"] = umap_result[:, 1]
	save_path = '../fam_plots'
	save_name = 'FAM_TSNE_' + features_types[current_idx] + '_' + str(current_distance) + '_' + str(out_dim) + '.png'
	save_path = os.path.join(save_path, save_name)

	palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(fam_values)))
	plt.figure(figsize=(16,10))
	sns.scatterplot(x="comp-1", y="comp-2", hue="Fam",
                data=df, palette=palette).set(title="T-SNE projection")

	plt.savefig(save_path) 
	print ('TSNE plot saved in %s' %save_name)

	save_path = '../fam_plots'
	save_name = 'FAM_PCA_' + features_types[current_idx] + '_' + str(current_distance) + '_' + str(out_dim) + '.png'
	save_path = os.path.join(save_path, save_name)

	palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(fam_values)))
	plt.figure(figsize=(16,10))
	sns.scatterplot(x="pca-one", y="pca-two", hue="Fam",
                data=df, palette=palette).set(title="PCA projection")

	plt.savefig(save_path)
	print ('PCA plot saved in %s' %save_name)

	save_path = '../fam_plots'
	save_name = 'FAM_UMAP_' + features_types[current_idx] + '_' + str(current_distance) + '_' + str(out_dim) + '.png'
	save_path = os.path.join(save_path, save_name)

	palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(fam_values)))
	plt.figure(figsize=(16,10))
	sns.scatterplot(x="umap-1", y="umap-2", hue="Fam",
                data=df, palette=palette).set(title="UMAP projection")

	plt.savefig(save_path)
	print ('UMAP plot saved in %s' %save_name)
