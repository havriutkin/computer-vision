import numpy as np

from points import *

NUM_OF_CORRESP = 6
NUM_OF_SAMPLES = 100

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
    
    @staticmethod
    def get_random():
        rotation = random_so3_cayley()
        translation = random_point_unit_ball(dim=3)

        return Camera(rotation, translation)


def normalize(v):
    return v / np.linalg.norm(v)

def random_points(num: int) -> list[np.ndarray]:
    result = []
    for _ in range(num):
        result.append(random_point_unit_ball(dim=3))

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
    T = np.column_stack((r1, r2, r3))

    return T

def so3_to_cayley(R: np.ndarray) -> np.ndarray:
    I = np.eye(3)
    S = np.linalg.inv(I + R) @ (I - R)

    s1 = S[2, 1]
    s2 = S[0, 2]
    s3 = S[1, 0]

    return np.array([s1, s2, s3])

if __name__ == "__main__":
    data = []

    for _ in range(NUM_OF_SAMPLES):
        camera_1 = Camera.get_random()
        camera_2 = Camera.get_random()
        label = 1   # Assume it is solvable

        # Generate random points in the world and project them into the two cameras
        world_points = random_points(NUM_OF_CORRESP)
        X = []
        Y = []
        for point in world_points:
            x = camera_1.project(point)
            y = camera_2.project(point)

            X.append(x)
            Y.append(y)


        # Apply standardization to the points
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

            a = float(x_new[1] / x_new[2])
            b = float(y_new[1] / y_new[2])

            r_params = so3_to_cayley(r_new)
            s_params = so3_to_cayley(s_new)

            if (i != 0):
                data_point.extend([float(r) for r in r_params])
                data_point.extend([float(s) for s in s_params])
            data_point.extend([a, b])
        
        # With 50% chance, corrupt random number in data_point
        if np.random.rand() < 0.5:
            idx = np.random.randint(len(data_point))
            data_point[idx] = np.random.rand()
            label = 0

        data.append({
            "data_point": data_point,
            "label": label
        })

    point = data[0]
    print(f"Dimension of the data point: {len(point['data_point'])}")

    # Count number of labels:
    num_pos = sum([1 for point in data if point["label"] == 1])
    num_neg = sum([1 for point in data if point["label"] == 0])
    print(f"Number of positive samples: {num_pos}")
    print(f"Number of negative samples: {num_neg}")
