import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


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

EPOCHS = 100
BATCH = 32
LAYERS = 3

if __name__ == "__main__":
    # Split data into train and test
    dataset = JSONDataset("../data.json")
    print(f"Dataset size: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    from collections import Counter
    label_counts = Counter(int(label.item()) for _, label in dataset)
    print("Label distribution in dataset:", label_counts)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

    # Initialize model
    model = MyModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Stats to track 
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    gradient_norms = []
    precisions = []
    recalls = []
    f1_scores = []

    def accuracy(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean().item()
    
    # Train model
    for epoch in range(EPOCHS):
        model.train()

        epoch_loss = 0
        epoch_loss = 0
        correct = 0
        total = 0
        total_grad_norm = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            labels = labels.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_grad_norm += total_norm ** 0.5

            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            labels = labels.long()
            correct += (preds.squeeze(1) == labels.squeeze(1)).sum().item()

            total += labels.size(0)
        
        avg_loss = epoch_loss / total
        acc = correct / total
        gradient_norms.append(total_grad_norm / len(train_loader))
        train_losses.append(avg_loss)
        train_accuracies.append(acc)

        # Evaluation
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                labels = labels.unsqueeze(1).float()

                outputs = model(inputs)
                probs = torch.sigmoid(outputs)  # apply sigmoid to get probabilities  

                loss = criterion(outputs, labels)

                preds = (probs > 0.5).long()    

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy().flatten())

                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        if epoch % 10 == 0:
            print(f"\nEpoch {epoch+1}")
            print(f"  Train Loss:  {avg_loss:.4f}, Train Acc:  {acc:.4f}")
            print(f"  Val Loss:    {val_losses[-1]:.4f}, Val Acc:    {val_accuracies[-1]:.4f}")
            print(f"  Grad Norm:   {gradient_norms[-1]:.4f}")
            print(f"  Precision:   {precision:.4f}")
            print(f"  Recall:      {recall:.4f}")
            print(f"  F1 Score:    {f1:.4f}")


    # Save model
    #torch.save(model.state_dict(), "model.pth")
    #print("Model saved")

    # Visualize training loss
    print("Prediction distribution:", Counter(map(int, np.array(all_preds).flatten())))
    epochs = range(1, len(train_losses) + 1)

    plt.hist(all_probs, bins=50)
    plt.title("Sigmoid Output Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, precisions, label='Precision')
    plt.plot(epochs, recalls, label='Recall')
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.title("Precision / Recall / F1 over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, gradient_norms, label='Gradient Norm')
    plt.title("Gradient Norms")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.show()

    # ROC curve
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)

    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()

    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()

