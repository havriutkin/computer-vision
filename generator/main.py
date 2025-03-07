import numpy as np

from points import *

NUM_OF_CORRESP = 6
NUM_OF_SAMPLES = 100

def normalize(v):
    return v / np.linalg.norm(v)

def random_points(num: int) -> list[np.ndarray]:
    result = []
    for _ in range(num):
        result.append(random_point_unit_ball(dim=3))

    return result

def random_camera() -> np.ndarray:
    rotation = random_so3_cayley()
    translation = random_point_unit_ball(dim=3)

    return np.hstack((rotation, translation[:, np.newaxis]))

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

    # Generate solvable samples
    for i in range(NUM_OF_SAMPLES // 2):
        points = random_points(NUM_OF_CORRESP)
        camera_1 = random_camera()
        camera_2 = random_camera()

        x = []
        y = []

        for point in points:
            extended_point = np.hstack((point, 1))
            projected_1 = camera_1 @ extended_point
            projected_2 = camera_2 @ extended_point

            # Homogenize
            projected_1 /= projected_1[-1]
            projected_2 /= projected_2[-1]

            x.append(projected_1)
            y.append(projected_2)

        transforms_x: list[np.ndarray] = []
        transforms_y: list[np.ndarray] = []
        for j in range(NUM_OF_CORRESP // 2):
            R = special_transform(x[j], x[j + 1])
            S = special_transform(y[j], y[j + 1])

            transforms_x.append(R)
            transforms_y.append(S)
        
        features = []
        for j in range(len(transforms_x)):
            R = transforms_x[j].T @ transforms_x[0]
            S = transforms_y[0].T @ transforms_y[j]
            x_new = transforms_x[j].T @ x[j + 1]
            y_new = transforms_y[j].T @ y[j + 1]
            a = float(x_new[1] / x_new[2])
            b = float(y_new[1] / y_new[2])


            R_params = so3_to_cayley(R).tolist()
            S_params = so3_to_cayley(S).tolist()

            if j != 0:
                features.extend(R_params)
                features.extend(S_params)
            features.extend([a, b])
        
        data.append({
            "features": features,
            "label": 1
        })

    # Generate unsolvable samples
    for i in range(NUM_OF_SAMPLES // 2):
        points = random_points(NUM_OF_CORRESP)
        camera_1 = random_camera()
        camera_2 = random_camera()

        x = []
        y = []

        for point in points:
            extended_point = np.hstack((point, 1))
            projected_1 = camera_1 @ extended_point
            projected_2 = random_point_unit_ball(dim=3)

            # Homogenize
            projected_1 /= projected_1[-1]
            projected_2 /= projected_2[-1]

            x.append(projected_1)
            y.append(projected_2)

        transforms_x: list[np.ndarray] = []
        transforms_y: list[np.ndarray] = []
        for j in range(NUM_OF_CORRESP // 2):
            R = special_transform(x[j], x[j + 1])
            S = special_transform(y[j], y[j + 1])

            transforms_x.append(R)
            transforms_y.append(S)
        
        features = []
        for j in range(len(transforms_x)):
            R = transforms_x[j].T @ transforms_x[0]
            S = transforms_y[0].T @ transforms_y[j]

            x_new = transforms_x[j].T @ x[j + 1]
            y_new = transforms_y[j].T @ y[j + 1]
            a = float(x_new[1] / x_new[2])
            b = float(y_new[1] / y_new[2])

            R_params = so3_to_cayley(R).tolist()
            S_params = so3_to_cayley(S).tolist()

            if j != 0:
                features.extend(R_params)
                features.extend(S_params)
            features.extend([a, b])
        
        data.append({
            "features": features,
            "label": 0
        })

    print(data[:5])
    point = data[0]
    print(f"Dimension of the feature vector: {len(point['features'])}")
