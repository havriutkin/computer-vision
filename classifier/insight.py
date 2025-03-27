from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import json
import torch
from torch.utils.data import Dataset


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
    
dataset = JSONDataset("data.json")

# Extract features and labels
X = np.array([sample["data_point"] for sample in dataset.data])
y = np.array([sample["label"] for sample in dataset.data])

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2D
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection")
plt.colorbar(label="Label")
plt.show()
