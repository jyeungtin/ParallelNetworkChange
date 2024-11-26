
## preliminaries ======================================================================================================

## libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
import networkx as nx
from tqdm import tqdm
import random

## temporal GNN set-up =================================================================================================

class TemporalGNN (nn.Module):

    ## main function
    def __init__(self, num_nodes, feature_dim=64, num_heads=4):
        super(TemporalGNN, self).__init__()
        
        # node embedding layer
        self.node_embedding = nn.Embedding (num_nodes, feature_dim)
        
        # temporal embedding layer
        self.temporal_embedding = nn.Linear (1, feature_dim)
        
        # graph attn layers
        self.gat1 = GATConv (feature_dim, feature_dim, heads = num_heads, concat = True)
        self.gat2 = GATConv (feature_dim * num_heads, feature_dim, heads = 1, concat = False)
        
        # final pred layer
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    ## forward function
    def forward (self, x, edge_index, edge_temporal_features):

        # transforming node indices into embeddings
        x = self.node_embedding(x)
        
        # gen temporal embeddings
        temporal_emb = self.temporal_embedding (edge_temporal_features.unsqueeze(1))
        
        # combine node embeddings with temp embeddings for involved nodes
        row, col = edge_index
        x_i = x[row] + temporal_emb
        x_j = x[col] + temporal_emb
        
        # graph attn convolutions
        x_i = self.gat1 (x_i, edge_index)
        x_i = nn.functional.relu(x_i)
        x_i = self.gat2 (x_i, edge_index)
        
        # generate edge-level preds
        edge_features = torch.cat([x_i[row], x_i[col]], dim=1)
        
        return self.predictor(edge_features).squeeze()

## data prep function
def prepare_temporal_dataset (df):
    
    # creating node mapping
    unique_nodes = np.unique (np.concatenate ([df['ori_node'], df['des_node']]))
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    
    # normalisning minutes using MinMaxScaler so thast they're between 0 and 1
    minute_scaler = MinMaxScaler()
    normalized_minutes = minute_scaler.fit_transform (df['minutes'].values.reshape(-1, 1)).flatten()
    
    # reformat graph to PyTorch Geometric format
    edge_index = torch.tensor([
        [node_to_idx[row['ori_node']] for _, row in df.iterrows()],
        [node_to_idx[row['des_node']] for _, row in df.iterrows()]
    ], dtype=torch.long)
    
    # use node indices as the features
    num_nodes = len(unique_nodes)
    x = torch.arange (num_nodes, dtype = torch.long)
    
    # temporal features
    temporal_features = torch.tensor (normalized_minutes, dtype = torch.float)
    
    # edge weights
    edge_weights = torch.tensor (df['km'].values, dtype = torch.float)
    
    return {
        'x': x,
        'edge_index': edge_index,
        'edge_weights': edge_weights,
        'temporal_features': temporal_features,
        'num_nodes': num_nodes,
        'node_mapping': node_to_idx
    }

## negative edge generation
def generate_negative_edges (data, num_negative_edges):

    # set of existing edges
    existing_edges = set (zip (data['edge_index'][0].numpy(), data['edge_index'][1].numpy()))
    
    # generate negative edges
    negative_edges = []
    while len(negative_edges) < num_negative_edges:

        # randomly sample nodes
        src = np.random.randint(0, data['num_nodes'])
        dst = np.random.randint(0, data['num_nodes'])
        
        # make sure there are no self-loops and no existing edges
        if src != dst and (src, dst) not in existing_edges and (dst, src) not in existing_edges:
            negative_edges.append((src, dst))
    
    return torch.tensor(negative_edges, dtype=torch.long).t()

## training function
def train_temporal_gnn (df, epochs = 100, learning_rate = 0.001):

    # prepping dataset
    print("🔍 Preparing temporal graph dataset...")
    data = prepare_temporal_dataset(df)
    
    # generating negative edges for link prediction
    negative_edges = generate_negative_edges (data, num_negative_edges = len(data['edge_index'][0]) // 2)
    
    # combining positive and negative edges
    full_edge_index = torch.cat([data['edge_index'], negative_edges], dim = 1)
    
    # creating edge labels (1 for existing edges, 0 for negative edges)
    edge_labels = torch.cat([
        torch.ones(data['edge_index'].shape[1]),
        torch.zeros(negative_edges.shape[1])
    ])
    
    # creating a matching temporal features tensor
    temporal_features = torch.cat([
        data['temporal_features'], 
        torch.rand(negative_edges.shape[1]) * data['temporal_features'].max()
    ])
    
    # initialise model
    print("🚀 Initialising Temporal GNN model...")
    model = TemporalGNN (num_nodes = data['num_nodes'])
    optimizer = optim.Adam (model.parameters(), lr = learning_rate)
    criterion = nn.BCELoss()
    
    # training loop with progress tracking
    max_auc = 0
    print("🏋️ Starting training...")
    for epoch in tqdm(range(epochs), desc="Training Epochs"):

        model.train()
        optimizer.zero_grad()
        
        # forward pass
        predictions = model(
            data['x'], 
            full_edge_index, 
            temporal_features
        )
        
        # loss computation
        loss = criterion (predictions, edge_labels)
        loss.backward()
        optimizer.step()
        
        # AUC computation
        with torch.no_grad():
            auc = roc_auc_score (edge_labels.numpy(), predictions.numpy())
            max_auc = max (max_auc, auc)
        
        # periodic progress reporting
        if (epoch + 1) % 10 == 0:
            print(f"🎯 Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}, Current AUC = {auc:.4f}, Max AUC = {max_auc:.4f}")
    
    print(f"🏆 You ate! Training Complete Slay!")
    return max_auc

def set_seed(seed):
    """
    Set random seed for reproducibility (I've added a bunch of scenarios if we end up running on multiple GPUs for example)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False