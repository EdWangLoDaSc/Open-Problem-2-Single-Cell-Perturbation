from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch_geometric.data import Data

# Split node_features
node_features_train, node_features_val = train_test_split(
    node_features, test_size=0.2, random_state=42)

# Split labels
labels_train, labels_val = train_test_split(
    labels, test_size=0.2, random_state=42)

# Split train_feature
train_feature_train, train_feature_val = train_test_split(
    train_feature, test_size=0.2, random_state=42)
node_features_train = torch.tensor(node_features_train, dtype=torch.float)
train_feature_train = torch.tensor(train_feature_train, dtype=torch.float)
labels_train = torch.tensor(labels_train, dtype=torch.float)
node_features_val = torch.tensor(node_features_val, dtype=torch.float)
train_feature_val = torch.tensor(train_feature_val, dtype=torch.float)
labels_val = torch.tensor(labels_val, dtype=torch.float)
node_features_to = torch.tensor(node_features, dtype=torch.float)

data_train = Data(x=node_features_to, edge_index=new_edge_index, global_features=train_feature_train, y=labels_train)
data_val = Data(x=node_features_to, edge_index=new_edge_index, global_features=train_feature_val, y=labels_val)
model = GNNModule(num_node_features=1,num_global_features = 390, num_output=18211)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(data_train)
    loss = criterion(output, labels_train)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Training loss: {loss.item()}')

    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(data_val)
        val_loss = criterion(val_output, labels_val)
        val_mse = nn.MSELoss()(val_output, labels_val)

        print(f'Epoch {epoch + 1}, Validation loss: {val_loss.item()}, Validation MSE: {val_mse.item()}')
