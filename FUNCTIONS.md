## Protein Graph Generation

### Description
This script converts protein files in PDB format into graph representations. Each amino acid is represented as a node, and the edges represent the distance between them. The resulting graphs are saved as PyTorch Geometric data objects.

### Dependencies
- numpy==1.21.5
- blosum (custom module)
- warnings

### Functions

#### `_get_sequence(structure)`
- Description: Extracts the amino acid sequence from a protein structure.
- Inputs:
  - `structure`: Protein structure object.
- Returns:
  - `sequence`: Amino acid sequence as a string.

#### `_get_structure(file)`
- Description: Parses the protein file and returns the protein structure object.
- Inputs:
  - `file`: Path to the protein file.
- Returns:
  - `structure`: Protein structure object.

#### `_get_adjacency(file)`
- Description: Generates the adjacency matrix representing the distances between amino acids in a protein.
- Inputs:
  - `file`: Path to the protein file.
- Returns:
  - `m`: Adjacency matrix as a dense numpy array.

#### `_get_edgeindex(file)`
- Description: Generates the edge index tensor representing the edges between amino acids in a protein graph.
- Inputs:
  - `file`: Path to the protein file.
- Returns:
  - `edge_index`: Edge index tensor as a torch tensor.

#### `_get_one_hot_symbftrs(sequence)`
- Description: Generates the one-hot encoded symbol features for each amino acid in the sequence.
- Inputs:
  - `sequence`: Amino acid sequence as a string.
- Returns:
  - `one_hot_symb`: One-hot encoded symbol features as a torch tensor.

#### `_get_res_ftrs(sequence)`
- Description: Generates the predefined residue features for each amino acid in the sequence.
- Inputs:
  - `sequence`: Amino acid sequence as a string.
- Returns:
  - `res_ftrs_out`: Residue features as a torch tensor.

#### `_get_blosum_fts(sequence, d)`
- Description: Generates the BLOSUM features for each amino acid in the sequence.
- Inputs:
  - `sequence`: Amino acid sequence as a string.
  - `d`: Size of the BLOSUM features.
- Returns:
  - `blos_matr`: BLOSUM features as a torch tensor.

#### `_get_SeqVec_ftrs(sequence)`
- Description: Generates the SeqVec features for each amino acid in the sequence.
- Inputs:
  - `sequence`: Amino acid sequence as a string.
- Returns:
  - `protein_embd`: SeqVec features as a torch tensor.

#### `_get_BERT_ftrs(sequence)`
- Description: Generates the BERT features for each amino acid in the sequence.
- Inputs:
  - `sequence`: Amino acid sequence as a string.
- Returns:
  - `embedding`: BERT features as a torch tensor.

#### `train(model, device, trainloader, optimizer, epoch)`
- Description: Trains the specified model using the given data loader and optimizer.
- Inputs:
  - `model`: Model to be trained.
  - `device`: Device to be used (CPU or GPU).
  - `trainloader`: Data loader for the training data.
  - `optimizer`: Optimizer for model parameter updates.
  - `epoch`: Current epoch number.

#### `predict(model, device, loader)`
- Description: Performs prediction using the trained model on the given data loader.
- Inputs:
  - `model`: Trained model for prediction.
  - `device`: Device to be used (CPU or GPU).
  - `loader`: Data loader for the test data.
- Returns:
  - `labels`: True labels of the test samples.
  - `predictions`: Predicted distances between protein pairs.

## Training and Evaluation

### Dependencies
- models (custom module)
- torch_optimizer==0.1.1
- numpy==1.21.5
- matplotlib==3.5.1
- tqdm==4.62.3
- pathlib
- math
- sklearn
- torch
- torch.nn
- torch.optim.lr_scheduler

### Functions

#### `get_mse(actual, predicted)`
- Description: Calculates the mean squared error (MSE) between the actual and predicted distances.
- Inputs:
  - `actual`: True distances between protein pairs.
  - `predicted`: Predicted distances between protein pairs.
- Returns:
  - `loss`: MSE loss.

#### `train(model, device, trainloader, optimizer, epoch)`
- Description: Trains the model using the given training data.
- Inputs:
  - `model`: Model to be trained.
  - `device`: Device to be used (CPU or GPU).
  - `trainloader`: Data loader for the training data.
  - `optimizer`: Optimizer for model parameter updates.
  - `epoch`: Current epoch number.

#### `predict(model, device, loader)`
- Description: Performs prediction using the trained model on the given data loader.
- Inputs:
  - `model`: Trained model for prediction.
  - `device`: Device to be used (CPU or GPU).
  - `loader`: Data loader for the test data.
- Returns:
  - `labels`: True labels of the test samples.
  - `predictions`: Predicted distances between protein pairs.

## Dataset Preparation

### Dependencies
- prepare_dataset (custom module)
- torch
- numpy==1.21.5
- pandas==1.3.5

### Classes

#### `LabelledDataset`
- Description: Custom dataset class for loading labeled protein data.
- Inputs:
  - `csv_file`: Path to the CSV file containing protein pair labels.
  - `processed_path`: Path to the directory containing processed protein graph data.
- Methods:
  - `__getitem__(self, index)`: Retrieves the protein graphs and label at the specified index.
  - `__len__(self)`: Returns the total number of samples in the dataset.

#### `SimpleDataset`
- Description: Custom dataset class for loading protein embeddings and labels.
- Inputs:
  - `embeddings`: Protein embeddings as numpy array or torch tensor.
  - `labels`: Labels corresponding to the protein embeddings.
- Methods:
  - `__getitem__(self, index)`: Retrieves the embedding and label at the specified index.
  - `__len__(self)`: Returns the total number of samples in the dataset.

## Main Training and Evaluation

### Dependencies
- models (custom module)
- torch
- numpy==1.21.5
- matplotlib==3.5.1
- tqdm==4.62.3
- pathlib
- math
- sklearn
- torch_optimizer==0.1.1

### Functions

#### `train(model, device, trainloader, optimizer, epoch)`
- Description: Trains the model using the given training data and optimizer.
- Inputs:
  - `model`: Model to be trained.
  - `device`: Device to be used (CPU or GPU).
  - `trainloader`: Data loader for the training data.
  - `optimizer`: Optimizer for model parameter updates.
  - `epoch`: Current epoch number.

#### `predict(model, device, loader)`
- Description: Performs prediction using the trained model on the given data loader.
- Inputs:
  - `model`: Trained model for prediction.
  - `device`: Device to be used (CPU or GPU).
  - `loader`: Data loader for the test data.
- Returns:
  - `labels`: True labels of the test samples.
  - `predictions`: Predicted distances between protein pairs.

## Model Selection and Training Loop

### Dependencies
- models (custom module)
- torch
- numpy==1.21.5
- matplotlib==3.5.1
- tqdm==4.62.3
- pathlib
- math
- sklearn
- torch.optim
- torch.nn
- torch.optim.lr_scheduler

### Constants
- `DISTANCE_TORCH`: Dictionary mapping distance function names to corresponding functions.

### Functions

#### `get_mse(actual, predicted)`
- Description: Calculates the mean squared error (MSE) between the actual and predicted distances.
- Inputs:
  - `actual`: True distances between protein pairs.
  - `predicted`: Predicted distances between protein pairs.
- Returns:
  - `loss`: MSE loss.

#### `train(model, device, trainloader, optimizer, epoch)`
- Description: Trains the model using the given training data, optimizer, and scheduler.
- Inputs:
  - `model`: Model to be trained.
  - `device`: Device to be used (CPU or GPU).
  - `trainloader`: Data loader for the training data.
  - `optimizer`: Optimizer for model parameter updates.
  - `epoch`: Current epoch number.

#### `predict(model, device, loader)`
- Description: Performs prediction using the trained model on the given data loader.
- Inputs:
  - `model`: Trained model for prediction.
  - `device`: Device to be used (CPU or GPU).
  - `loader`: Data loader for the test data.
- Returns:
  - `labels`: True labels of the test samples.
  - `predictions`: Predicted distances between protein pairs.

## Execution

### Dependencies
- models (custom module)
- torch
- numpy==1.21.5
- matplotlib==3.5.1
- tqdm==4.62.3
- pathlib
- math
- sklearn
- torch.optim
- torch.nn
- torch.optim.lr_scheduler

### Constants
- `processed_path`: Path to the directory containing processed protein graph data.
- `csv_file`: Path to the CSV file containing protein pair labels.
- `current_distance`: Name of the distance function to be used.
- `out_dim`: Output dimension of the model.
- `in_dim`: Input dimension of the model.
- `num_epochs`: Number of epochs for training.
- `min_loss`: Minimum loss achieved during training.
- `n_epochs_stop`: Number of epochs to wait before early stopping.

### Execution
- Load the labeled dataset.
- Split the dataset into training and test sets.
- Create data loaders for training and test data.
- Initialize the model and move it to the specified device.
- Initialize the optimizer.
- Set the minimum loss to a high value.
- Train the model for the specified number of epochs.
- Perform prediction on the test data.
- Calculate the loss.
- Save the model if the loss is the lowest achieved so far.
- Apply early stopping if the loss does not improve for a specified number of epochs.

## Conclusion

This project focuses on generating protein graph representations and training a model for protein structure comparison. The code provides functions for protein graph generation, model training, and evaluation. The provided datasets and models can be easily customized and extended for different protein comparison tasks. The project aims to contribute to the field of bioinformatics and enable more efficient protein structure analysis and classification.

For more details, please refer to the code documentation and comments within the code.
