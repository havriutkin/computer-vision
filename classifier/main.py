import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
import json
import pandas as pd
import matplotlib.pyplot as plt

class JSONDataset(Dataset):
    """ JSON Dataset """
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        data_points = torch.tensor(sample["data_point"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return data_points, label

class MyModel(nn.Module):
    """ Binary Classification NN """
    def __init__(self):
        super(MyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

EPOCHS = 200
BATCH = 64
LAYERS = 3

if __name__ == "__main__":
    # Split data into train and test
    dataset = JSONDataset("../data.json")
    print(f"Dataset size: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

    # Initialize model
    model = MyModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    train_loses = []    # for visualization
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            labels = labels.unsqueeze(1).float()

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
        for inputs, labels in test_loader:
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            predicted = (probs >= 0.5).float()

            y_true += labels.squeeze().tolist()
            y_pred += predicted.squeeze().tolist()

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

