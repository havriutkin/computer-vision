import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os

# Load preprocessed data
df = pd.read_csv("data.csv")
data = df.values
X = data[:, :-1].astype(np.float32)
y = data[:, -1].astype(np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create DataLoaders
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)

# Define MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#cTraining loop with metrics
num_epochs = 20
history = {'train_loss': [], 'val_loss': []}

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    history['train_loss'].append(epoch_loss)
    
    # validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(test_loader.dataset)
    history['val_loss'].append(val_loss)
    
    print(f"Epoch {epoch}/{num_epochs} - train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}")

# Plot and save training curves
plt.figure()
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Validation'])
plt.savefig('/mnt/data/loss_curve.png')

# ROC and Precision-Recall curves
model.eval()
y_scores = []
y_true = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        y_scores.append(model(xb).cpu().numpy())
        y_true.append(yb.numpy())
y_scores = np.concatenate(y_scores)
y_true = np.concatenate(y_true)

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr)
plt.title(f'ROC Curve (AUC = {roc_auc:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('/mnt/data/roc_curve.png')

precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision)
plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('/mnt/data/pr_curve.png')

# Save model state_dict and TorchScript module
os.makedirs('/mnt/data/model', exist_ok=True)
torch.save(model.state_dict(), '/mnt/data/model/model_state.pth')
scripted = torch.jit.script(model.cpu())
scripted.save('/mnt/data/model/model_scripted.pt')

print("Training curves saved to /mnt/data/loss_curve.png")
print("ROC curve saved to /mnt/data/roc_curve.png")
print("PR curve saved to /mnt/data/pr_curve.png")
print("Model state_dict: /mnt/data/model/model_state.pth")
print("Scripted model: /mnt/data/model/model_scripted.pt")
