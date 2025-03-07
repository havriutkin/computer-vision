import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

# Input points (in homogeneous coordinates)
p1 = np.array([3, 6, 1.0])  # Replace x1 and y1 with numbers.
p2 = np.array([10, 4, 1.0])  # Replace x2 and y2 with numbers.

r3 = normalize(p1)               # Ensures that p1 lies on the z-axis => has coordinates [0, 0, ||p1||]
r1 = normalize(np.cross(p1, p2)) # Ensures that x-axis is perpendicular to p1 and p2 => x coordinate is 0
r2 = np.cross(r3, r1)            # Finish the orthonormal basis
T = np.vstack((r1, r2, r3))

# Test points
Tp1 = T @ p1
Tp2 = T @ p2

print(f"T = \n{T}")

print("T p1 =", Tp1)
print("T p2 =", Tp2)
