import numpy as np
from sympy import Matrix
import json

from points import *

NUM_OF_CORRESP = 6
NUM_OF_SAMPLES = 10000

def isSO3(R: np.ndarray) -> bool:
    orthogonality = np.allclose(R @ R.T, np.eye(3))
    det = np.isclose(np.linalg.det(R), 1)

    return orthogonality and det


class Camera:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        assert rotation.shape == (3, 3)
        assert translation.shape == (3,)
        assert isSO3(rotation)

        self.rotation = rotation
        self.translation = translation
        self.projection_matrix = np.hstack((rotation, translation.reshape(3, 1)))

    def project(self, point: np.ndarray) -> np.ndarray:
        assert point.shape == (3,)

        extended_point = np.hstack((point, 1))
        projected = self.projection_matrix @ extended_point
        dehomogenized = projected / projected[-1]

        return dehomogenized
    
    def get_kernel(self):
        matr = Matrix(self.projection_matrix)
        kernel = matr.nullspace()
        kernel = np.array(kernel[0]).astype(np.float64)

        return kernel
    
    @staticmethod
    def get_random():
        
        flag = False
        while not flag:
            rotation = random_so3_cayley()
            translation = random_point_on_ball(dim=3, radius=5)

            c = Camera(rotation, translation)

            #todo: test it
            # Condition 1
            kernel = c.get_kernel()
            kernel /= kernel[-1]
            kernel = kernel[:2]
            #print(kernel)
            norm1 = np.linalg.norm(kernel)

            # Condition 2
            translation_dehom = c.translation / c.translation[-1]
            translation_dehom = translation_dehom[:2]
            norm2 = np.linalg.norm(translation_dehom)

            if norm1 > 2 and norm2 < 1:
                flag = True

        return c

    @staticmethod
    def get_simple_camera_pair(d: float = 3.0):
        """ Return camera [R | t] such that R = I, t = [0, 0, -d]"""
        identity = np.eye(3)
        translation = np.array([0, 0, -d])
        camera_1 = Camera(identity, translation)

        rotation = random_so3_cayley()
        camera_2 = Camera(rotation, translation)

        return camera_1, camera_2

def normalize(v):
    return v / np.linalg.norm(v)

def random_points(num: int, radius: float = 1) -> list[np.ndarray]:
    result = []
    for _ in range(num):
        result.append(random_point_on_ball(dim=3, radius=radius))

    return result

def essential_matrix(camera_1: np.ndarray, camera_2: np.ndarray) -> np.ndarray:
    R1 = camera_1[:, :3]
    t1 = camera_1[:, 3]

    R2 = camera_2[:, :3]
    t2 = camera_2[:, 3]

    t = t2 - t1
    skew_t = np.array([[0, -t[2], t[1]],
                       [t[2], 0, -t[0]],
                       [-t[1], t[0], 0]])
    
    return R2 @ skew_t @ R1.T

def special_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    r3 = normalize(p1)
    r1 = normalize(np.cross(p1, p2))
    r2 = np.cross(r3, r1)
    T = np.vstack((r1, r2, r3))

    return T

def so3_to_cayley(R: np.ndarray) -> np.ndarray:
    I = np.eye(3)
    S = np.linalg.inv(I + R) @ (I - R)

    s1 = S[2, 1]
    s2 = S[0, 2]
    s3 = S[1, 0]

    s = np.array([s1, s2, s3])

    return s

if __name__ == "__main__":
    data = []

    print("Generating data...")
    for _ in range(NUM_OF_SAMPLES):
        if _ % 100 == 0:
            print(f"Generating sample {_}...")
        camera_1, camera_2 = Camera.get_simple_camera_pair(d=3.0)
        label = 1   # Assume it is solvable

        # Generate random points in the world and project them into the two cameras
        world_points = random_points(NUM_OF_CORRESP)
        X = []
        Y = []
        for point in world_points:
            x = camera_1.project(point)
            y = camera_2.project(point) 
            if np.random.rand() < 0.1:
                label = 0
                if np.random.rand() < 0.5:
                    x = random_point_inside_ball(dim=3, radius=10)
                    x[2] = 1
                else:
                    y = random_point_inside_ball(dim=3, radius=10)
                    y[2] = 1

            X.append(x)
            Y.append(y)

        # Find encoding transformations to the points
        R = []
        S = []
        for i in range(NUM_OF_CORRESP - 1):
            x1, x2 = X[i], X[i + 1]
            y1, y2 = Y[i], Y[i + 1]

            r = special_transform(x1, x2)
            s = special_transform(y1, y2)

            R.append(r)
            S.append(s)

        # Convert data point to the proper format
        data_point = []
        for i in range(NUM_OF_CORRESP // 2):
            r_new = R[i].T @ R[0]
            s_new = S[0].T @ S[i]

            x_new = R[i].T @ X[i + 1]
            y_new = S[i].T @ Y[i + 1]
            
            # Dehomogenization
            a = float(x_new[1] / x_new[2])
            b = float(y_new[1] / y_new[2])

            r_params = so3_to_cayley(r_new)
            s_params = so3_to_cayley(s_new)

            if (i != 0):
                data_point.extend([float(r) for r in r_params])
                data_point.extend([float(s) for s in s_params])
            data_point.extend([a, b])

        data.append({
            "data_point": data_point,
            "label": label,
        })
    print("Data generation completed.")

    # === Normalize a and b across dataset ===
    """
    print("Normalizing (a, b) values...")
    all_a = []
    all_b = []
    ab_indices = [(0, 1), (8, 9), (16, 17)]

    for point in data:
        for a_idx, b_idx in ab_indices:
            all_a.append(point["data_point"][a_idx])
            all_b.append(point["data_point"][b_idx])

    mean_a, std_a = np.mean(all_a), np.std(all_a)
    mean_b, std_b = np.mean(all_b), np.std(all_b)

    for point in data:
        for a_idx, b_idx in ab_indices:
            point["data_point"][a_idx] = (point["data_point"][a_idx] - mean_a) / (std_a + 1e-8)
            point["data_point"][b_idx] = (point["data_point"][b_idx] - mean_b) / (std_b + 1e-8)

    print("Normalization completed.")
    print(f"Mean a: {mean_a:.4f}, std a: {std_a:.4f}")
    print(f"Mean b: {mean_b:.4f}, std b: {std_b:.4f}")
    """

    # print("Applying corruption to 50% of samples...")
    # for point in data:
    #     if np.random.rand() < 0.5:
    #         idx = np.random.randint(len(point["data_point"]))
    #         point["data_point"][idx] = random_point_inside_ball(dim=1, radius=10)[0]
    #         point["label"] = 0

    # Count number of labels
    num_pos = sum(1 for point in data if point["label"] == 1)
    num_neg = sum(1 for point in data if point["label"] == 0)
    print(f"Number of positive samples: {num_pos}")
    print(f"Number of negative samples: {num_neg}")

    # Save to file
    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)

