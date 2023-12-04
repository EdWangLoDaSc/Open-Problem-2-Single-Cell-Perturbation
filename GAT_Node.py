# --- Import Libraries ---
import pandas as pd
import numpy as np
import networkx as nx
import anndata
import tensorflow as tf
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.sparse import coo_matrix
import torch
import dask.dataframe as dd

# --- Data Loading and Processing ---
de_train = pd.read_parquet("/content/drive/MyDrive/de_train.parquet")
gene_expression = de_train.iloc[:, 5:]

# Load Parquet file
adata_train = dd.read_parquet('/content/drive/MyDrive/adata_train.parquet')

# Extract 'gene' and 'normalized_count' information
gene_normalized_count = adata_train[['gene', 'normalized_count']]
result = gene_normalized_count.compute()
gene_info_df = result.groupby('gene')['normalized_count'].mean().reset_index()

# --- Graph Creation and Analysis ---
def create_graph_from_correlation_matrix(correlation_matrix, threshold=0.99):
    mask = np.abs(correlation_matrix) > threshold
    np.fill_diagonal(mask, False)
    row, col = np.where(mask)
    data = correlation_matrix[row, col]
    sparse_matrix = coo_matrix((data, (row, col)), shape=correlation_matrix.shape)
    G = nx.from_scipy_sparse_array(sparse_matrix, create_using=nx.Graph, edge_attribute='weight')
    return G

# Create AnnData object
adata = anndata.AnnData(X=gene_expression)

# Calculate adjacency matrix (Pearson correlation)
adjacency_matrix = np.corrcoef(adata.X, rowvar=False)

# Create graph from correlation matrix
G = create_graph_from_correlation_matrix(adjacency_matrix)

# Add gene names as node attributes
genes = gene_expression.columns.tolist()
for i, gene in enumerate(genes):
    G.nodes[i]["name"] = gene

# Add normalized counts as node attributes
for node in G.nodes(data=True):
    node_id, node_data = node
    gene_name = node_data['name']
    normalized_count = gene_info_df[gene_info_df['gene'] == gene_name]['normalized_count'].values
    G.nodes[node_id]['normalized_count'] = normalized_count[0] if normalized_count.size > 0 else None

# Remove nodes without edges
nodes_with_no_edges = [node for node, degree in G.degree() if degree == 0]
G.remove_nodes_from(nodes_with_no_edges)

# Remove nodes with low expression
threshold = 4.7
low_expression_nodes = [node for node, node_data in G.nodes(data=True) 
                        if node_data['normalized_count'] is not None and node_data['normalized_count'] < threshold]
G.remove_nodes_from(low_expression_nodes)

# --- Feature Extraction ---
def extract_node_features(G):
    normalized_counts = np.array([
        G.nodes[node_id]['normalized_count'] for node_id in sorted(G.nodes())
    ])
    normalized_counts = np.nan_to_num(normalized_counts, nan=0.0)
    node_features = normalized_counts.reshape(-1, 1)
    return node_features

node_features = extract_node_features(G)
edge_list = np.array(G.edges()).T
edge_index = torch.tensor(edge_list, dtype=torch.long)
train_feature = torch.tensor(node_features, dtype=torch.float)

# Assign labels
labels = torch.tensor(np.array(de_train.iloc[:,5:]), dtype=torch.float)

# --- Update Edge Index ---
def update_edge_index(edge_index, num_nodes):
    max_index = edge_index.max()
    present_nodes = torch.unique(edge_index)
    mapping = -torch.ones(max_index + 1, dtype=torch.long)
    mapping[present_nodes] = torch.arange(num_nodes)
    new_edge_index = mapping[edge_index]
    mask = (new_edge_index[0] != -1) & (new_edge_index[1] != -1)
    new_edge_index = new_edge_index[:, mask]
    return new_edge_index

num_nodes = len(G.nodes)  # Update with the correct number of nodes
new_edge_index = update_edge_index(edge_index, num_nodes)

