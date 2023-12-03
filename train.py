from utils import *
from sklearn.cluster import KMeans
import copy
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import pickle
import argparse
from models import CustomTransformer_v3  # Can be changed to other models in models.py
import os


def train_epoch(model, dataloader, optimizer, criterion, device='cuda'):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, val_dataloader, criterion, label_reducer=None, scaler=None, device='cuda'):
    model.eval()
    val_loss = 0.0
    val_predictions_list = []
    val_targets_list = []
    with torch.no_grad():
        for val_inputs, val_targets in val_dataloader:
            val_targets_list.append(val_targets.clone().cpu())
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_predictions = model(val_inputs)
            if label_reducer:
                val_targets = torch.tensor(
                    label_reducer.transform(scaler.transform(val_targets.clone().cpu().detach().numpy())),
                    dtype=torch.float32).to(device)
            val_loss += criterion(val_predictions, val_targets).item()
            val_predictions_list.append(val_predictions.cpu())

    val_loss /= len(val_dataloader)

    val_predictions_stacked = torch.cat(val_predictions_list, dim=0)
    val_targets_stacked = torch.cat(val_targets_list, dim=0)

    return val_loss, val_targets_stacked, val_predictions_stacked


def validate_sampling_strategy(sampling_strategy):
    allowed_strategies = ['k-means', 'random']
    if sampling_strategy not in allowed_strategies:
        raise ValueError(f"Invalid sampling strategy. Choose from: {', '.join(allowed_strategies)}")


def train_func(X_train, Y_reduced, X_val, Y_val, n_components, num_epochs, batch_size, label_reducer, scaler,
               d_model=128, early_stopping=5000, device='cuda', ):
    best_mrrmse = float('inf')
    best_model = None
    best_val_loss = float('inf')
    best_epoch = 0
    # model = CustomTransformer(num_features=X_train.shape[1], num_labels=n_components, d_model=d_model).to(device)
    model = CustomTransformer_v3(num_features=X_train.shape[1], num_labels=n_components, d_model=d_model).to(device)
    # model = CustomDeeperModel(X_train.shape[1], d_model, n_components).to(device)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                            torch.tensor(Y_reduced, dtype=torch.float32).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device),
                                              torch.tensor(
                                                  Y_val,
                                                  dtype=torch.float32).to(device)),
                                batch_size=batch_size, shuffle=False)
    if n_components < 18211:
        lr = 1e-3

    else:
        lr = 1e-5
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7, verbose=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9999, patience=500,
                                               verbose=True)
    criterion = nn.HuberLoss()
    # criterion = nn.L1Loss()
    # criterion = CustomLoss()
    # criterion = nn.MSELoss()
    model.train()
    counter = 0
    pbar = tqdm(range(num_epochs), position=0, leave=True)
    for epoch in range(num_epochs):
        _ = train_epoch(model, dataloader, optimizer, criterion)

        if counter >= early_stopping:
            break
        if scaler:
            val_loss, val_targets_stacked, val_predictions_stacked = validate(model, val_dataloader, criterion,
                                                                              label_reducer, scaler)
            # Calculate MRRMSE for the entire validation set
            val_mrrmse = calculate_mrrmse_np(
                val_targets_stacked.cpu().detach().numpy(),
                scaler.inverse_transform((label_reducer.inverse_transform(
                    val_predictions_stacked.cpu().detach().numpy()))))
        else:
            val_loss, val_targets_stacked, val_predictions_stacked = validate(model, val_dataloader, criterion)
            val_mrrmse = calculate_mrrmse_np(val_targets_stacked.cpu().detach().numpy(),

                                             val_predictions_stacked.cpu().detach().numpy())

        if val_mrrmse < best_mrrmse:
            best_mrrmse = val_mrrmse
            # best_model = copy.deepcopy(model)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
            best_epoch = epoch
        else:
            counter += 1

        pbar.set_description(
            f"Validation best MRRMSE: {best_mrrmse:.4f} Validation best loss:"
            f" {best_val_loss:.4f} Last epoch: {best_epoch}")
        pbar.update(1)
        # scheduler.step()  # for cosine anealing
        scheduler.step(val_loss)
    return label_reducer, scaler, best_model


def train_transformer_k_means_learning(X, Y, n_components, num_epochs, batch_size,
                                       d_model=128, early_stopping=5000, device='cuda', seed=18):
    label_reducer, scaler, Y_reduced = reduce_labels(Y, n_components)
    Y_reduced = Y_reduced.to_numpy()
    Y = Y.to_numpy()
    num_clusters = 2
    validation_percentage = 0.1

    # Create a K-Means clustering model
    kmeans = KMeans(n_clusters=num_clusters, n_init=100)

    # Fit the model to your regression targets (Y)
    clusters = kmeans.fit_predict(Y)

    # Initialize lists to store the training and validation data
    X_train, Y_train = [], []
    X_val, Y_val = [], []

    # Iterate through each cluster
    for cluster_id in range(num_clusters):
        # Find the indices of data points in the current cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        print(len(cluster_indices))
        if len(cluster_indices) >= 20:
            # Split the data points in the cluster into training and validation
            train_indices, val_indices = train_test_split(cluster_indices, test_size=validation_percentage,
                                                          random_state=seed)

            # Append the corresponding data points to the training and validation sets
            X_train.extend(X[train_indices])
            Y_train.extend(Y_reduced[train_indices])  # Y_reduced for train Y for validation
            X_val.extend(X[val_indices])
            Y_val.extend(Y[val_indices])
        else:
            X_train.extend(X[cluster_indices])
            Y_train.extend(Y_reduced[cluster_indices])  # Y_reduced for train Y for validation
    # Convert the lists to numpy arrays if needed
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    transfromer_model = train_func(X_train, Y_train, X_val, Y_val, n_components, num_epochs, batch_size,
                                   label_reducer, scaler, d_model, early_stopping, device)

    return label_reducer, scaler, transfromer_model


def train_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                           early_stopping, batch_size, device):
    # Training loop for k_means sampling strategy
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, transformer_model = train_transformer_k_means_learning(
                one_hot_encode_features,
                targets,
                n_components,
                num_epochs=num_epochs,
                early_stopping=early_stopping,
                batch_size=batch_size,
                d_model=d_model, device=device)
            os.makedirs('trained_models_random', exist_ok=True)
            # Save the trained models
            with open(f'trained_models_random/label_reducer_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(label_reducer, file)

            with open(f'trained_models_random/scaler_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(scaler, file)

            torch.save(transformer_model.state_dict(),
                       f'trained_models_random/transformer_model_{n_components}_{d_model}.pt')


def train_non_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                               early_stopping, batch_size, device, seed, validation_percentage):
    # Split the data for non-k_means sampling strategy
    X_train, X_val, y_train, y_val = split_data(one_hot_encode_features, targets, test_size=validation_percentage,
                                                shuffle=True, random_state=seed)

    # Training loop for non-k_means sampling strategy
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, Y_reduced = reduce_labels(y_train, n_components)
            transformer_model = train_func(X_train, y_train, X_val, y_val,
                                           n_components,
                                           num_epochs=num_epochs,
                                           early_stopping=early_stopping,
                                           batch_size=batch_size,
                                           d_model=d_model,
                                           label_reducer=label_reducer,
                                           scaler=scaler,
                                           device=device)

            # Save the trained models
            os.makedirs('trained_models_k-means', exist_ok=True)
            with open(f'trained_models_k-means/label_reducer_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(label_reducer, file)

            with open(f'trained_models_k-means/scaler_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(scaler, file)

            torch.save(transformer_model.state_dict(),
                       f'trained_models_k-means/transformer_model_{n_components}_{d_model}.pt')


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument('--config', type=str, help="Path to the YAML config file.", default='config_train.yaml')
    args = parser.parse_args()

    # Check if the config file is provided
    if not args.config:
        print("Please provide a config file using --config.")
        return

    # Load and print configurations
    config_file = args.config
    config = load_and_print_config(config_file)

    # Access specific values from the config
    n_components_list = config.get('n_components_list', [])
    d_models_list = config.get('d_models_list', [])  # embedding dimensions for the transformer models
    batch_size = config.get('batch_size', 32)
    sampling_strategy = config.get('sampling_strategy', 'random')
    data_file = config.get('data_file', '')
    id_map_file = config.get('id_map_file', '')
    validation_percentage = config.get('validation_percentage', 0.2)
    device = config.get('device', 'cuda')
    seed = config.get('seed', None)
    num_epochs = config.get('num_epochs', 20000)
    early_stopping = config.get('early_stopping', 5000)

    # Validate the sampling strategy
    validate_sampling_strategy(sampling_strategy)

    # Prepare augmented data
    one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(data_file=data_file,
                                                                            id_map_file=id_map_file)

    if sampling_strategy == 'k-means':
        train_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                               early_stopping, batch_size, device)
    else:
        train_non_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                                   early_stopping, batch_size, device, seed, validation_percentage)


if __name__ == "__main__":
    main()
