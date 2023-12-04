from IPython.display import clear_output as clr
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
chemberta._modules["lm_head"] = nn.Identity()
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
chemberta.eval()
def featurize_ChemBERTa(smiles_list, padding=True):
    embeddings_cls = torch.zeros(len(smiles_list), 384)
    embeddings_mean = torch.zeros(len(smiles_list), 384)

    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True)
            model_output = chemberta(**encoded_input)

            embedding = model_output[0][::,0,::]
            embeddings_cls[i] = embedding

            embedding = torch.mean(model_output[0],1)
            embeddings_mean[i] = embedding




class GNNModule(nn.Module):
    def __init__(self, num_node_features, num_global_features, num_output):
        super(GNNModule, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc_global = nn.Linear(num_global_features, 64)
        self.fc_final = nn.Linear(32 + 64, num_output)

    def forward(self, data):
        x, edge_index, global_features, batch = data.x, data.edge_index, data.global_features, data.batch

        # Node features
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Global features (per graph)
        #unique_batch_indices = data.batch.unique(sorted=True)
        #global_features_expanded = global_features[unique_batch_indices]

        # Expanding global features to match the number of nodes
        #global_features_expanded = global_features_expanded[batch]

        # Global features
        global_features_expanded = self.fc_global(global_features)
        global_features_expanded = F.relu(global_features_expanded)
        print(x.shape)
        print(global_features_expanded.shape)
        # Concatenate node features and global features
        x = torch.cat([x, global_features_expanded], dim=1)

        # Final prediction
        x = self.fc_final(x)
        return x

    return embeddings_cls.numpy(), embeddings_mean.numpy()


class GNNModule(nn.Module):
    def __init__(self, num_node_features, num_global_features, num_output):
        super(GNNModule, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc_global = nn.Linear(num_global_features, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc_final = nn.Linear(384, num_output)  # Adjusted input feature size

    def forward(self, data):
        # Node features
        x, edge_index, global_features = data.x, data.edge_index, data.global_features

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.fc1(x)
        x = self.fc2(x)
        # Expanding graph features to match the number of global features
        graph_features = x.repeat(global_features.size(0), 1)

        # Global features
        global_features = self.fc_global(global_features)
        global_features = F.relu(global_features)
        global_features = self.fc3(global_features)
        global_features = F.relu(global_features)
        global_features = self.fc4(global_features)
        global_features = F.relu(global_features)

        # Concatenating graph features and global features
        combined_features = torch.cat([graph_features, global_features], dim=1)
        output = self.fc_final(combined_features)
        output = F.relu(output)
        print(output.shape)
        return output


clr()
feat_train = de_train[['cell_type', 'sm_name', 'SMILES', ]]
sm_name2smiles = {
        name: smiles
        for name, smiles
        in feat_train.drop_duplicates(subset='sm_name').iloc[::,1:].values
    }

#id_map['SMILES'] = [sm_name2smiles[name] for name in id_map

                       #.sm_name.values]

train_cls_pad_true, train_mean_pad_true = featurize_ChemBERTa(de_train.SMILES)
#test_cls_pad_true, test_mean_pad_true = featurize_ChemBERTa(id_map.SMILES)
