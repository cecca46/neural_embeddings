import numpy as np
import blosum as bl
import warnings
warnings.filterwarnings("ignore")
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib
import networkx as nx
import torch
import biographs as bg
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from torch_geometric.data import Data
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder

# list of 20 amminoacids
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

#Dictionary for getting Residue symbols
ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN':'Q', 'ARG':'R', 'SER': 'S','THR': 'T', 'VAL': 'V', 'TRP':'W', 'TYR': 'Y'}

# residue features stored as key-value pair
pcp_dict = {'A':[ 0.62014, -0.18875, -1.2387, -0.083627, -1.3296, -1.3817, -0.44118],
            'C':[0.29007, -0.44041,-0.76847, -1.05, -0.4893, -0.77494, -1.1148],
            'D':[-0.9002, 1.5729, -0.89497, 1.7376, -0.72498, -0.50189, -0.91814],
            'E':[-0.74017, 1.5729, -0.28998, 1.4774, -0.25361, 0.094051, -0.4471],
            'F':[1.1903, -1.1954, 1.1812, -1.1615, 1.1707, 0.8872, 0.02584],
            'G':[ 0.48011, 0.062916, -1.9949, 0.25088, -1.8009, -2.0318, 2.2022],
            'H':[-0.40009, -0.18875, 0.17751, 0.77123, 0.5559, 0.44728, -0.71617],
            'I':[1.3803, -0.84308, 0.57625, -1.1615, 0.10503, -0.018637, -0.21903],
            'K':[-1.5003, 1.5729, 0.75499, 1.1057, 0.44318, 0.95221, -0.27937],
            'L':[1.0602, -0.84308, 0.57625, -1.273, 0.10503, 0.24358, 0.24301],
            'M':[0.64014, -0.59141, 0.59275, -0.97565, 0.46368, 0.46679, -0.51046],
            'N':[-0.78018, 1.0696, -0.38073, 1.2172, -0.42781, -0.35453, -0.46879],
            'P':[0.12003, 0.062916, -0.84272, -0.1208, -0.45855, -0.75977, 3.1323],
            'Q':[-0.85019, 0.16358, 0.22426, 0.8084, 0.04355, 0.24575, 0.20516],
            'R':[-2.5306, 1.5729, 0.89249, 0.8084, 1.181, 1.6067, 0.11866],
            'S':[-0.18004, 0.21392, -1.1892, 0.32522, -1.1656, -1.1282, -0.48056],
            'T':[-0.050011, -0.13842, -0.58422, 0.10221, -0.69424, -0.63625, -0.50017],
            'V':[1.0802, -0.69208, -0.028737, -0.90132, -0.36633, -0.3762, 0.32502],
            'W':[0.81018, -1.6484, 2.0062, -1.0872, 2.3901, 1.8299, 0.032377],
            'Y':[0.26006, -1.0947, 1.2307, -0.78981, 1.2527, 1.1906, -0.18876]}

class ProteinGraphConverter:
    def __init__(self, processed_dir='../processed_graphs/lstm_ds', proteins_dir='../data/pdb_files/'):
        """
        Converter class to convert protein files in PDB format to graph representation.

        Args:
        processed_dir (str): Path to the directory to save the processed graph data.
        proteins_dir (str): Path to the directory containing the protein files in PDB format.
        """
        self.processed_dir = processed_dir
        self.proteins_dir = proteins_dir
        self.protein_files = [f for f in listdir(proteins_dir) if isfile(join(proteins_dir, f))]
        self.embedder = SeqVecEmbedder()

    def _get_sequence(self, structure):
        """
        Extract the amino acid sequence from the PDB structure.

        Args:
        structure (Bio.PDB.Structure.Structure): PDB structure.

        Returns:
        str: Amino acid sequence.
        """
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in ressymbl.keys():
                        sequence = sequence + ressymbl[residue.get_resname()]
        return sequence

    def _get_structure(self, file):
        """
        Parse the PDB file and get the structure.

        Args:
        file (str): Path to the PDB file.

        Returns:
        Bio.PDB.Structure.Structure: PDB structure.
        """
        parser = PDBParser()
        structure = parser.get_structure("id", file)
        return structure

    def _get_adjacency(self, file):
        """
        Generate the adjacency matrix from the PDB file.

        Args:
        file (str): Path to the PDB file.

        Returns:
        numpy.ndarray: Adjacency matrix.
        """
        molecule = bg.Pmolecule(file)
        network = molecule.network()
        mat = nx.adjacency_matrix(network)
        m = mat.todense()
        if deepSet:
            m = np.identity(m.shape[0])
        return m

    def _get_edgeindex(self, file):
        """
        Generate the edge index from the adjacency matrix.

        Args:
        file (str): Path to the PDB file.

        Returns:
        torch.Tensor: Edge index.
        """
        edge_ind = []
        m = self._get_adjacency(file)
        
        a = np.nonzero(m > 0)[0]
        b = np.nonzero(m > 0)[1]
        
        edge_ind.append(a)
        edge_ind.append(b)
        edge_ind = np.array(edge_ind)

        return torch.tensor(edge_ind, dtype=torch.long)

    def _get_one_hot_symbftrs(self, sequence):
        """
        Generate one-hot encoded symbol features from the amino acid sequence.

        Args:
        sequence (str): Amino acid sequence.

        Returns:
        torch.Tensor: One-hot encoded symbol features.
        """
        one_hot_symb = np.zeros((len(sequence), len(pro_res_table)))
        row = 0
        for res in sequence:
            col = pro_res_table.index(res)
            one_hot_symb[row][col] = 1
            row += 1
        return torch.tensor(one_hot_symb, dtype=torch.float)

    def _get_res_ftrs(self, sequence):
        """
        Generate residue features from the amino acid sequence.

        Args:
        sequence (str): Amino acid sequence.

        Returns:
        torch.Tensor: Residue features.
        """
        res_ftrs_out = []
        for res in sequence:
            res_ftrs_out.append(pcp_dict[res])
        res_ftrs_out = np.array(res_ftrs_out)
        return torch.tensor(res_ftrs_out, dtype=torch.float)

    def _get_blosum_fts(self, sequence, d=25):
        """
        Generate BLOSUM features from the amino acid sequence.

        Args:
        sequence (str): Amino acid sequence.
        d (int): Dimension of the BLOSUM features.

        Returns:
        torch.Tensor: BLOSUM features.
        """
        matrix = bl.BLOSUM(62)
        blos_matr = np.zeros((len(sequence), d))
        row = 0
        for res in sequence:
            val = matrix[res]
            blos_matr[row] = np.fromiter(val.values(), dtype=float)
            row += 1
        return torch.tensor(blos_matr, dtype=torch.float)

    def _get_SeqVec_ftrs(self, sequence):
        """
        Generate SeqVec features from the amino acid sequence.

        Args:
        sequence (str): Amino acid sequence.

        Returns:
        torch.Tensor: SeqVec features.
        """
        embedding = self.embedder.embed(sequence)
        protein_embd = torch.tensor(embedding).sum(dim=0)
        return torch.tensor(protein_embd, dtype=torch.float)

    def _get_BERT_ftrs(self, sequence):
        """
        Generate ProtTrans BERT features from the amino acid sequence.

        Args:
        sequence (str): Amino acid sequence.

        Returns:
        torch.Tensor: ProtTrans BERT features.
        """
        embedder = ProtTransBertBFDEmbedder()
        embedding = embedder.embed(sequence)
        return torch.tensor(embedding, dtype=torch.float)

    def convert_files(self):
        """
        Convert protein files to graph representation and save them as PyTorch Geometric data objects.
        """
        for file_name in tqdm(self.protein_files):
            try:
                full_path = os.path.join(self.proteins_dir, file_name)
                struct = self._get_structure(full_path)
                seq = self._get_sequence(struct)
                # node_feats = self._get_one_hot_symbftrs(seq)
                # node_feats = self._get_res_ftrs(seq)
                node_feats = self._get_SeqVec_ftrs(seq)
                # node_feats = self._get_BERT_ftrs(seq)
                # node_feats = self._get_blosum_fts(seq)
                edge_index = self._get_edgeindex(full_path)
                mat = self._get_adjacency(full_path)
                if node_feats.shape[0] != mat.shape[0]:
                    print('Feature matrix shape is', node_feats.shape)
                    print('Adjacency matrix shape is', mat.shape)
                    break
                data = Data(x=node_feats, edge_index=edge_index)
                save_path = os.path.join(self.processed_dir, file_name.split('.')[0] + '.pt')
                torch.save(data, save_path)
            except:
                print('Skipping file: %s' % file_name)
