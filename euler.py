import math
import numpy as np

def rotation_matrix_to_z_axis(x1, y1):
    """
    Returns the 3x3 rotation matrix (no scaling) that rotates
    [x1, y1, 1] onto [0, 0, sqrt(x1^2 + y1^2 + 1]].
    """
    # Step 1: compute angles
    alpha = math.atan2(y1, x1)         # direction in XY-plane
    r = math.sqrt(x1**2 + y1**2)
    phi = math.atan(r)                 # angle w.r.t. z-axis

    # Yaw (around z) = -alpha
    psi = -alpha
    # Pitch (around y) = -phi
    theta = -phi
    # Roll (around x) = 0
    # Build up rotation R_y(theta) * R_z(psi)

    Rz_psi = np.array([
        [ math.cos(psi), -math.sin(psi), 0],
        [ math.sin(psi),  math.cos(psi), 0],
        [ 0            ,  0            , 1]
    ])

    Ry_theta = np.array([
        [ math.cos(theta),  0, math.sin(theta)],
        [ 0              ,  1, 0             ],
        [-math.sin(theta),  0, math.cos(theta)]
    ])

    R_total = Ry_theta @ Rz_psi  # matrix multiplication
    return R_total

# Example usage:
x1, y1 = 2.0, 3.0

R = rotation_matrix_to_z_axis(x1, y1)
p_original = np.array([x1, y1, 1.0])
p_rotated = R @ p_original

print("Original point: ", p_original)
print("Rotated point:  ", p_rotated)
