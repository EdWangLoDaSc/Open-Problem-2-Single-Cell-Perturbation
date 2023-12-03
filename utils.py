import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim
from lion_pytorch import Lion
from sklearn.preprocessing import StandardScaler
import yaml
import pickle
from torch.utils.data import TensorDataset, DataLoader


# Evaluate the loaded model on the test data
def evaluate_model(model, dataloader, criterion=None):
    model.eval()
    total_output = []
    total_labels = []
    running_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            total_output.append(outputs)
            total_labels.append(labels)
            num_batches += 1
        total_mrrmse = calculate_mrrmse(torch.cat(total_labels, dim=0), torch.cat(total_output, dim=0))
    if not criterion:
        return total_mrrmse.detach().cpu().item()
    return total_mrrmse.detach().cpu().item(), running_loss / num_batches


def load_transformer_model(n_components, input_features, d_model, models_foler='trained_models', device='cuda'):
    # transformer_model = CustomTransformer(num_features=input_features, num_labels=n_components, d_model=d_model).to(
    #     device)
    transformer_model = CustomTransformer_v3(num_features=input_features, num_labels=n_components, d_model=d_model).to(
        device)
    # transformer_model = CustomDeeperModel(input_features, d_model, n_components).to(device)
    transformer_model.load_state_dict(torch.load(f'trained_models/transformer_model_{n_components}_{d_model}.pt'))
    transformer_model.eval()
    if n_components == 18211:
        return None, None, transformer_model
    label_reducer = pickle.load(open(f'{models_foler}/label_reducer_{n_components}_{d_model}.pkl', 'rb'))
    scaler = pickle.load(open(f'{models_foler}/scaler_{n_components}_{d_model}.pkl', 'rb'))
    return label_reducer, scaler, transformer_model


def reduce_labels(Y, n_components):
    if n_components == Y.shape[1]:
        return None, None, Y
    label_reducer = TruncatedSVD(n_components=n_components, n_iter=10)
    scaler = StandardScaler()

    Y_scaled = scaler.fit_transform(Y)
    Y_reduced = label_reducer.fit_transform(Y_scaled)

    return label_reducer, scaler, Y_reduced


def prepare_augmented_data(
        data_file="",
        id_map_file=""):
    de_train = pd.read_parquet(data_file)
    id_map = pd.read_csv(id_map_file)
    xlist = ['cell_type', 'sm_name']
    _ylist = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    y = de_train.drop(columns=_ylist)

    # Combine train and test data for one-hot encoding
    combined_data = pd.concat([de_train[xlist], id_map[xlist]])

    dum_data = pd.get_dummies(combined_data, columns=xlist)

    # Split the combined data back into train and test
    train = dum_data.iloc[:len(de_train)]
    test = dum_data.iloc[len(de_train):]
    # uncommon = [f for f in train if f not in test]
    # X = train.drop(columns=uncommon)
    X = train
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]

    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    std_cell_type = de_cell_type.groupby('cell_type').std().reset_index()

    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    std_sm_name = de_sm_name.groupby('sm_name').std().reset_index()

    # Append mean and std for 'cell_type'
    rows = []
    for name in de_cell_type['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        std_rows = std_cell_type[std_cell_type['cell_type'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    tr_cell_type = pd.concat(rows)
    tr_cell_type = tr_cell_type.reset_index(drop=True)

    # Append mean and std for 'sm_name'
    rows = []
    for name in de_sm_name['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        std_rows = std_sm_name[std_sm_name['sm_name'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    tr_sm_name = pd.concat(rows)
    tr_sm_name = tr_sm_name.reset_index(drop=True)

    # Similar process for test data
    rows = []
    for name in id_map['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        std_rows = std_cell_type[std_cell_type['cell_type'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    te_cell_type = pd.concat(rows)
    te_cell_type = te_cell_type.reset_index(drop=True)

    rows = []
    for name in id_map['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        std_rows = std_sm_name[std_sm_name['sm_name'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    te_sm_name = pd.concat(rows)
    te_sm_name = te_sm_name.reset_index(drop=True)

    # Join mean and std features to X0, test0
    X0 = X.join(tr_cell_type.iloc[:, 1:]).copy()
    X0 = X0.join(tr_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    # Remove string columns
    X0 = X0.select_dtypes(exclude='object')

    y0 = y.iloc[:, :].copy()

    test0 = test.join(te_cell_type.iloc[:, 1:]).copy()
    test0 = test0.join(te_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    # Remove string columns
    test0 = test0.select_dtypes(exclude='object')
    return X0.astype(np.float32).to_numpy(), y0, test0.astype(np.float32).to_numpy()


def load_and_print_config(config_file):
    # Load configurations from the YAML file
    config = load_config(config_file)

    # Print loaded configurations
    print("Configurations:")
    print(config)

    return config


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def split_data(train_features, targets, test_size=0.3, shuffle=False, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(train_features, targets, test_size=test_size,
                                                      shuffle=shuffle, random_state=random_state)

    return X_train, X_val, y_train.to_numpy(), y_val.to_numpy()


def calculate_mrrmse_np(outputs, labels):
    # Calculate the Root Mean Squared Error (RMSE) row-wise
    rmse_per_row = np.sqrt(np.mean((outputs - labels.reshape(-1, outputs.shape[1])) ** 2, axis=1))
    # Calculate the Mean RMSE (MRMSE) across all rows
    mrmse = np.mean(rmse_per_row)
    return mrmse


# Function to calculate MRRMSE
def calculate_mrrmse(outputs, labels):
    # Calculate the Root Mean Squared Error (RMSE) row-wise
    rmse_per_row = torch.sqrt(torch.mean((outputs - labels) ** 2, dim=1))
    # Calculate the Mean RMSE (MRMSE) across all rows
    mrmse = torch.mean(rmse_per_row)
    return mrmse


# Custom mrrmse Loss Function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        loss = calculate_mrrmse(predictions, targets)  # + torch.abs(predictions-targets).mean()
        return loss.mean()


# Plot Loss and mrrmse
def plot_mrrmse(val_mrrmse):
    epochs = range(1, len(val_mrrmse) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_mrrmse, 'r', label='Validation mrrmse')
    plt.title('Validation mrrmse')
    plt.xlabel('Epochs')
    plt.ylabel('mrrmse')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Save Model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


# Model Architecture
class CustomTransformer(nn.Module):
    def __init__(self, num_features, num_labels, d_model=128, num_heads=8, num_layers=6):  # num_heads=8
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, d_model)
        # Embedding layer for sparse features
        # self.embedding = nn.Embedding(num_features, d_model)

        # self.norm = nn.BatchNorm1d(d_model, affine=True)
        self.norm = nn.LayerNorm(d_model)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
        #                                 dropout=0.1, device='cuda')
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, device='cuda', dropout=0.3,
                                       activation=nn.GELU(),
                                       batch_first=True), enable_nested_tensor=True, num_layers=num_layers
        )
        # Dropout layer for regularization
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        x = self.embedding(x)

        # x = (self.transformer(x,x))
        x = self.transformer(x)
        x = self.norm(x)
        # x = self.fc(self.dropout(x))
        x = self.fc(x)
        return x


class CustomTransformer_v3(nn.Module):  # mean + std
    def __init__(self, num_features, num_labels, d_model=128, num_heads=8, num_layers=6, dropout=0.3):
        super(CustomTransformer_v3, self).__init__()
        self.num_target_encodings = 18211 * 4
        self.num_sparse_features = num_features - self.num_target_encodings

        self.sparse_feature_embedding = nn.Linear(self.num_sparse_features, d_model)
        self.target_encoding_embedding = nn.Linear(self.num_target_encodings, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.concatenation_layer = nn.Linear(2 * d_model, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, activation=nn.GELU(),
                                       batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        sparse_features = x[:, :self.num_sparse_features]
        target_encodings = x[:, self.num_sparse_features:]

        sparse_features = self.sparse_feature_embedding(sparse_features)
        target_encodings = self.target_encoding_embedding(target_encodings)

        combined_features = torch.cat((sparse_features, target_encodings), dim=1)
        combined_features = self.concatenation_layer(combined_features)
        combined_features = self.norm(combined_features)

        x = self.transformer(combined_features)
        x = self.norm(x)

        x = self.fc(x)
        return x


class CustomTransformer_V2(nn.Module):  # v2 mean only
    def __init__(self, num_features, num_labels, d_model=128, num_heads=8, num_layers=6, dropout=0.1):
        super(CustomTransformer_V2, self).__init__()
        self.num_target_encodings = 18211  # *2 #each one for each type (cell or drug)
        self.num_sparse_features = num_features - 2 * self.num_target_encodings

        self.target_encoding_embedding_a = nn.Linear(self.num_target_encodings, d_model)
        self.target_encoding_embedding_b = nn.Linear(self.num_target_encodings, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.concatenation_layer = nn.Linear((self.num_sparse_features + 2 * d_model), d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                       dropout=dropout, activation=nn.GELU(), batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        # No need for embedding or mean pooling for one-hot encoded sparse features
        # Simply concatenate the one-hot encoded sparse features with target_encodings
        x = torch.cat(
            (x[:, :self.num_sparse_features],
             self.target_encoding_embedding_a(
                 x[:, self.num_sparse_features:self.num_target_encodings + self.num_sparse_features]),
             self.target_encoding_embedding_b(x[:, self.num_target_encodings + self.num_sparse_features:])), dim=1)
        x = self.concatenation_layer(x)
        x = self.norm(x)

        x = self.transformer(x)
        x = self.norm(x)

        x = self.fc(x)
        return x


class CustomDeeperModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, dropout=0.3, layer_norm=True):
        super(CustomDeeperModel, self).__init__()
        layers = []

        for _ in range(num_layers):
            if layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


# For evaluation
def train_and_evaluate_model(X_train, y_train, X_val, y_val, num_epochs, batch_size, learning_rate, val_loss_path):
    device = 'cuda'

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    min_val_mrrmse = float('inf')
    min_loss = float('inf')
    num_features = X_train.shape[1]
    num_labels = y_train.shape[1]

    model = CustomTransformer(num_features, num_labels).to(device)
    # criterion_mse = nn.MSELoss()
    # criterion_mae = nn.L1Loss()  # nn.HuberLoss()#  # Mean Absolute Error
    criterion_mae = nn.HuberLoss(reduction='sum')
    # criterion = CustomLoss()
    #
    # weight_decay = 1e-4
    betas = (0.9, 0.99)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-3, betas=betas)

    val_losses = []
    val_mrrmses = []
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.999)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0, verbose=True)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min",factor=0.9999, patience=250)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = criterion(outputs, targets)
            loss = criterion_mae(outputs, targets)  # criterion_mse(outputs,targets)#
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        # epoch_loss = running_loss / num_batches
        with torch.no_grad():
            val_mrrmse, epoch_loss = evaluate_model(model, val_loader, criterion_mae)
        if val_mrrmse < min_val_mrrmse:
            min_val_mrrmse = val_mrrmse
            save_model(model, 'mrrmse_val_' + val_loss_path)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            save_model(model, 'loss_val_' + val_loss_path)
        val_losses.append(epoch_loss)
        val_mrrmses.append(val_mrrmse)

        print(f'Epoch {epoch + 1}/{num_epochs} - Val MRRMSE: {val_mrrmse:.4f} - Loss: {epoch_loss:.4f}')
        # Adjust learning rate based on validation MRRMSE
        # scheduler.step(epoch_loss)
        scheduler.step()
    # Plot validation MRRMSE and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(val_mrrmses, label='Validation MRRMSE')
    plt.xlabel('Epoch')
    plt.ylabel('MRRMSE')
    plt.title('Validation MRRMSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.show()
