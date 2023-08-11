import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool as gep
from torch_geometric.nn import SAGEConv


class GCNN(nn.Module):
    def __init__(self, in_feat=20, out_feat=32, dropout=0.2):
        """
        GCNN (Graph Convolutional Neural Network) model.

        Args:
        in_feat (int): Input feature dimension. Defaults to 20.
        out_feat (int): Output feature dimension. Defaults to 32.
        dropout (float): Dropout rate. Defaults to 0.2.
        """
        super(GCNN, self).__init__()

        print("GCNN model instantiation")
        self.conv1 = GCNConv(in_feat, out_feat)
        self.conv2 = GCNConv(out_feat, out_feat)
        self.dropout = dropout

    def forward(self, protein_data):
        """
        Forward pass of the GCNN model.

        Args:
        protein_data (torch_geometric.data.Batch): Input protein data.

        Returns:
        torch.Tensor: Output of the GCNN model.
        """
        x, edge_index, batch = protein_data.x, protein_data.edge_index, protein_data.batch

        # Convolution + non-linearity
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convolution + non-linearity
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling
        x = gep(x, batch)

        return x


class GAT(nn.Module):
    def __init__(self, in_feat=20, out_feat=32, dropout=0.2, heads=4, concat=False):
        """
        GAT (Graph Attention Network) model.

        Args:
        in_feat (int): Input feature dimension. Defaults to 20.
        out_feat (int): Output feature dimension. Defaults to 32.
        dropout (float): Dropout rate. Defaults to 0.2.
        heads (int): Number of attention heads. Defaults to 4.
        concat (bool): Whether to concatenate the attention head outputs. Defaults to False.
        """
        super(GAT, self).__init__()

        print("GAT model instantiation")

        self.conv1 = GATConv(in_feat, out_feat, heads=heads, concat=concat)
        self.conv2 = GATConv(out_feat, out_feat, heads=heads, concat=concat)
        self.dropout = dropout

    def forward(self, protein_data):
        """
        Forward pass of the GAT model.

        Args:
        protein_data (torch_geometric.data.Batch): Input protein data.

        Returns:
        torch.Tensor: Output of the GAT model.
        """
        x, edge_index, batch = protein_data.x, protein_data.edge_index, protein_data.batch

        # Convolution + non-linearity
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convolution + non-linearity
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling
        x = gep(x, batch)

        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_feat=20, out_feat=32, dropout=0.2):
        """
        GraphSAGE (Graph Sample and Aggregation) model.

        Args:
        in_feat (int): Input feature dimension. Defaults to 20.
        out_feat (int): Output feature dimension. Defaults to 32.
        dropout (float): Dropout rate. Defaults to 0.2.
        """
        super(GraphSAGE, self).__init__()

        print("GraphSAGE model instantiation")
        self.conv1 = SAGEConv(in_feat, out_feat)
        self.conv2 = SAGEConv(out_feat, out_feat)
        self.dropout = dropout

    def forward(self, protein_data):
        """
        Forward pass of the GraphSAGE model.

        Args:
        protein_data (torch_geometric.data.Batch): Input protein data.

        Returns:
        torch.Tensor: Output of the GraphSAGE model.
        """
        x, edge_index, batch = protein_data.x, protein_data.edge_index, protein_data.batch

        # Convolution + non-linearity
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convolution + non-linearity
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling
        x = gep(x, batch)

        return x


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        """
        Multiclass Classification model.

        Args:
        num_feature (int): Number of input features.
        num_class (int): Number of classes.
        """
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(32)

    def forward(self, x):
        """
        Forward pass of the MulticlassClassification model.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output of the MulticlassClassification model.
        """
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x
