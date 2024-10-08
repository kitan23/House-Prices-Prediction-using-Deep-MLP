import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
from model_src import MLP
from data_src import load_and_preprocess_data


BATCH_SIZE = 128
LEARNING_RATE = 0.01

train_features, _ , train_labels = load_and_preprocess_data('data_src/train.csv', 'data_src/test.csv')

# split data into training and validation sets (80% train, 20% val)
# Play the ratio of train and validation sets to see if it improves the performance
train_size = int(0.8 * len(train_features))
val_size = len(train_features) - train_size
train_dataset, val_dataset = random_split(TensorDataset(train_features, train_labels), [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# loss
loss = torch.nn.MSELoss()

# model
net = MLP()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# optimizer with weight decay (L2 regularization)
# Play with different learning rate and weight decay to see if it improves the performance.
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0.01) # weigh_decay is L2 regularization

def log_rmse(net, data_loader):
    net.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            clipped_preds = torch.max(net(features), torch.tensor(1.0, device=device))  # clip preds to avoid log(0)
            loss = torch.sqrt(torch.mean((clipped_preds.log() - labels.log()) ** 2))
            total_loss += loss.item() * len(labels)
            count += len(labels)
    return total_loss / count


def calculate_mae(net, data_loader):
    net.eval()  
    total_loss = 0
    count = 0
    with torch.no_grad():  
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            predictions = net(features)
            loss = torch.mean(torch.abs(predictions - labels))  # Calculate absolute error
            total_loss += loss.item() * len(labels)  # Accumulate weighted loss
            count += len(labels)  

    return total_loss / count  


# Training loop
num_epochs = 500
train_ls, val_ls = [], []

train_mae_ls, val_mae_ls = [], []

# Initialize variables to track the best model and validation loss
best_val_rmse = float('inf')
best_model = None

# Training loop with model saving
for epoch in range(num_epochs):
    net.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = net(features)
        loss_val = loss(predictions, labels)
        loss_val.backward()
        optimizer.step()

    # Calculate RMSE for training and validation
    train_rmse = log_rmse(net, train_loader)
    val_rmse = log_rmse(net, val_loader)

    train_mae = calculate_mae(net, train_loader)
    val_mae = calculate_mae(net, val_loader)

    train_ls.append(train_rmse)
    val_ls.append(val_rmse)

    train_mae_ls.append(train_mae)
    val_mae_ls.append(val_mae)

    # Save the best model based on validation RMSE
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model = net.state_dict()  # Save model parameters

    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}, Train MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}')

# Save the best model to disk
torch.save(net.state_dict(), 'saved_models/best_mlp_house_price_model.pth')

print("Best model saved as 'best_mlp_house_price_model.pth' and predictions saved to 'submission.csv'.")


# Plot learning curves
# Function to plot the training and validation RMSE
def plot_rmse(train_ls, val_ls, num_epochs):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_ls, label='Train RMSE')
    plt.plot(range(1, num_epochs + 1), val_ls, label='Validation RMSE', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Train and Validation RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mae(train_mae_ls, val_mae_ls, num_epochs):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_mae_ls, label='Train MAE')
    plt.plot(range(1, num_epochs + 1), val_mae_ls, label='Validation MAE', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Train and Validation MAE')
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the function to plot RMSE
plot_rmse(train_ls, val_ls, num_epochs)

# Call the function to plot MAE
plot_mae(train_mae_ls, val_mae_ls, num_epochs)