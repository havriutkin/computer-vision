import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)

def straightening(p1: np.ndarray, p2: np.ndarray, eps: float = 0.01) -> np.ndarray:
    if np.linalg.norm(np.cross(p1, p2)) < eps:
        raise ValueError("Points are colinear")

    r3 = normalize(p1)
    r1 = normalize(np.cross(p1, p2))
    r2 = np.cross(r3, r1)
    T = np.vstack((r1, r2, r3))

    transformed = T.T @ p2
    encoding = float(transformed[1] / transformed[2])

    return T, encoding

def so3_to_cayley(R: np.ndarray) -> np.ndarray:
    I = np.eye(3)
    S = np.linalg.inv(I + R) @ (I - R)

    s1 = S[2, 1]
    s2 = S[0, 2]
    s3 = S[1, 0]

    return np.array([s1, s2, s3])

df = pd.read_csv("labeled_six_tuples.csv")
feature_cols = df.columns[2:-1]
X_flat = df[feature_cols].values
X = X_flat.reshape(-1, 6, 4)
y = df["label"].values

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["a_1","b_1",
              "r21_1","r21_2","r21_3","s12_1","s12_2","s12_3","a_2","b_2",
              "r31_1","r31_2","r31_3","s13_1","s13_2","s13_3","a_3","b_3","label"]
    writer.writerow(header)

    for i in tqdm(range(len(X)), desc="Preprocessing", unit="tuple"):
        corr = X[i]
        label = y[i]

        # Pick pairs 
        x1 = np.array([corr[0,0], corr[0,1], 1.0])
        x2 = np.array([corr[1,0], corr[1,1], 1.0])
        x3 = np.array([corr[2,0], corr[2,1], 1.0])
        x4 = np.array([corr[3,0], corr[3,1], 1.0])
        x5 = np.array([corr[4,0], corr[4,1], 1.0])
        x6 = np.array([corr[5,0], corr[5,1], 1.0])

        y1 = np.array([corr[0,2], corr[0,3], 1.0])
        y2 = np.array([corr[1,2], corr[1,3], 1.0])
        y3 = np.array([corr[2,2], corr[2,3], 1.0])
        y4 = np.array([corr[3,2], corr[3,3], 1.0])
        y5 = np.array([corr[4,2], corr[4,3], 1.0])
        y6 = np.array([corr[5,2], corr[5,3], 1.0])

        try:
            # Apply straightening
            R1, a1 = straightening(x1, x2)
            R2, a2 = straightening(x3, x4)
            R3, a3 = straightening(x5, x6)

            S1, b1 = straightening(y1, y2)
            S2, b2 = straightening(y3, y4)
            S3, b3 = straightening(y5, y6)
        except ValueError:
            tqdm.write(f"Tuple {i}: colinear points; skipping")
            continue

        r21 = so3_to_cayley(R2.T @ R1)
        r31 = so3_to_cayley(R3.T @ R1)
        s12 = so3_to_cayley(S1.T @ S2)
        s13 = so3_to_cayley(S1.T @ S3)

        row = [a1, b1,
               *r21, *s12, a2, b2,
               *r31, *s13, a3, b3,
               label]
        writer.writerow(row)

print("Done. Preprocessed data saved to data.csv")