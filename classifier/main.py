import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd
import matplotlib.pyplot as plt

class JSONDataset(Dataset):
    """ JSON Dataset """
    def __init__(self, json_path, scaler=StandardScaler()):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # Apply scaler to the data
        X = [sample["data_point"] for sample in self.data]
        X = scaler.fit_transform(X)
        for i, sample in enumerate(self.data):
            sample["data_point"] = X[i]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        data_points = torch.tensor(sample["data_point"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return data_points, label

class MyModel(nn.Module):
    """ Binary Classification NN """
    def __init__(self, hidden_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(18, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

EPOCHS = 500
BATCH = 64
LAYERS = 3

if __name__ == "__main__":
    # Split data into train and test
    dataset = JSONDataset("data.json")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

    # Initialize model
    model = nn.Sequential(
        nn.Linear(18, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_loses = []    # for visualization
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_loses.append(avg_epoch_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")

    # Evaluate model
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy}\n, Precision: {precision}\n, Recall: {recall}\n, F1: {f1}\n")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved")

    # Visualize training loss
    plt.plot(train_loses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    # Save plot
    plt.savefig(f"./visuals/Traing_Loss_{LAYERS}LRS_{EPOCHS}EP_{BATCH}_BTCH.png")

