from IPython.display import clear_output as clr
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn as nn

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

    return embeddings_cls.numpy(), embeddings_mean.numpy()
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
