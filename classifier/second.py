import torch
import torch.nn as nn
import torch.optim as optim 
import json
import copy
import numpy as np
import tqdm
from sklearn.model_selection import StratifiedKFold

def model_train(model: nn.Module, X_train, y_train, X_val, y_val):
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 250   
    batch_size = 32  
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    print("Training model")
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]

                # forward pass
                y_pred = model(X_batch)
                y_batch = y_batch.unsqueeze(1)
                loss = loss_fn(y_pred, y_batch)


                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()

                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc


if __name__ == '__main__':
    with open("data.json", "r") as f:
        data = json.load(f)

    print("Loading data")
    X = [sample["data_point"] for sample in data]
    X = torch.tensor(X, dtype=torch.float32)
    Y = [sample["label"] for sample in data]
    Y = torch.tensor(Y, dtype=torch.float32)

    class WideModel(nn.Module):
        def __init__(self):
            super(WideModel, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(18, 54),
                nn.ReLU(),
                nn.Linear(54, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)
        
    class DeepModel(nn.Module):
        def __init__(self):
            super(DeepModel, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(18, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    # Initialize models
    wide_model = WideModel()
    deep_model = DeepModel()

    # Use 5-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cv_scores_wide = []
    cv_scores_deep = []

    print("Starting Kfold cross validation")
    for train, test in kfold.split(X, Y):
        acc_w = model_train(wide_model, X[train], Y[train], X[test], Y[test])
        acc_d = model_train(deep_model, X[train], Y[train], X[test], Y[test])

        cv_scores_wide.append(acc_w)
        cv_scores_deep.append(acc_d)

    wide_acc = np.mean(cv_scores_wide)
    deep_acc = np.mean(cv_scores_deep)
    wide_std = np.std(cv_scores_wide)
    deep_std = np.std(cv_scores_deep)

    print(f"Wide model accuracy: {wide_acc:.4f} +/- {wide_std:.4f}")
    print(f"Deep model accuracy: {deep_acc:.4f} +/- {deep_std:.4f}")
    print("Done!")


