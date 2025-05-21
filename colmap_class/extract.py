import sqlite3
import numpy as np
import pycolmap
import itertools
import csv
import math
import random
from tqdm import tqdm

TOLERANCE = 1.0
MAX_ITERS = 10000   # max sextuple tests per image‐pair
MAX_PAIRS = 100

def decode_pair_id(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return int(image_id1), int(image_id2)

# 1) Load keypoints + raw matches
DB_PATH     = "../dataset/south-building/database.db"
conn        = sqlite3.connect(DB_PATH)
cursor      = conn.cursor()

keypoints = {}
cursor.execute("SELECT image_id, data FROM keypoints;")
for image_id, blob in cursor:
    keypoints[image_id] = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)

pair_matches = {}
cursor.execute("SELECT pair_id, data FROM matches WHERE data IS NOT NULL;")
for pair_id, blob in cursor:
    img1, img2 = decode_pair_id(pair_id)
    idxs       = np.frombuffer(blob, dtype=np.uint32).reshape(-1, 2)
    pts1       = keypoints[img1][idxs[:, 0]]
    pts2       = keypoints[img2][idxs[:, 1]]
    pair_matches[(img1, img2)] = (pts1, pts2)

cursor.close()
conn.close()
print(f"Number of image‐pairs: {len(pair_matches)}")

# 2) Sample up to MAX_PAIRS random image‐pairs
all_pairs = list(pair_matches.keys())
if len(all_pairs) > MAX_PAIRS:
    selected_pairs = random.sample(all_pairs, MAX_PAIRS)
else:
    selected_pairs = all_pairs
print(f"Processing {len(selected_pairs)} image‐pairs (random subset)")

# 2) Load the sparse reconstruction (for intrinsics)
recon = pycolmap.Reconstruction("../dataset/south-building/sparse")

# 3) Minimal‐solver options: disable RANSAC, single trial
minimal_opts = pycolmap.RANSACOptions(
    max_error=TOLERANCE,
    confidence=0.0,
    min_num_trials=1,
    max_num_trials=1,
    min_inlier_ratio=1.0
)

# 4) Prepare CSV output
csv_path = "labeled_six_tuples.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["img1","img2"]
    for i in range(6):
        header += [f"x1_{i}", f"y1_{i}", f"x2_{i}", f"y2_{i}"]
    header += ["label"]
    writer.writerow(header)

    # 5) Main loop with progress bar over image‐pairs
    for img_pair in tqdm(selected_pairs,
                         desc="Image pairs",
                         unit="pair"):
        img1, img2 = img_pair
        img1 = int(img1); img2 = int(img2)
        cam1 = recon.images[img1].camera
        cam2 = recon.images[img2].camera

        N = len(pts1)
        u1_h = np.column_stack([pts1, np.ones(N)])
        u2_h = np.column_stack([pts2, np.ones(N)])

        total_combos = math.comb(N, 6)
        # select tuples
        if total_combos <= MAX_ITERS:
            combos_to_test = list(itertools.combinations(range(N), 6))
        else:
            sampled = set()
            while len(sampled) < MAX_ITERS:
                combo = tuple(sorted(random.sample(range(N), 6)))
                sampled.add(combo)
            combos_to_test = list(sampled)

        labels = []

        # inner loop with progress bar over selected tuples
        for combo in tqdm(combos_to_test,
                          desc=f"Pair {img1}-{img2}",
                          total=len(combos_to_test),
                          leave=False,
                          unit="tuple"):

            pts1_6 = pts1[list(combo)]
            pts2_6 = pts2[list(combo)]

            solvable = False
            # test each 5‐subset
            for five_idxs in itertools.combinations(range(6), 5):
                sub1 = pts1_6[list(five_idxs)]
                sub2 = pts2_6[list(five_idxs)]

                result = pycolmap.estimate_essential_matrix(
                    sub1, sub2, cam1, cam2,
                    estimation_options=minimal_opts
                )
                if result is None:
                    continue

                E_cand = result["E"]
                idx6   = (set(range(6)) - set(five_idxs)).pop()
                u1_6   = np.array([*pts1_6[idx6], 1.0])
                u2_6   = np.array([*pts2_6[idx6], 1.0])
                err6   = abs(u2_6 @ E_cand @ u1_6)

                if err6 < TOLERANCE:
                    solvable = True
                    break

            label = 1 if solvable else 0
            labels.append((combo, label))

            # write row
            row = [img1, img2]
            for i in combo:
                x1, y1 = pts1[i]
                x2, y2 = pts2[i]
                row += [x1, y1, x2, y2]
            row += [label]
            writer.writerow(row)

        positives = sum(l for _, l in labels)
        negatives = len(labels) - positives
        tqdm.write(
            f"Pair ({img1},{img2}): tested {len(labels)}/{total_combos}, "
            f"solvable {positives}, not {negatives}"
        )

print(f"\nAll data written to {csv_path}")
