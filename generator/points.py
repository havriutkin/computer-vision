""" Methods for generating points on a unit Ball and elements of SO(3) group """
import numpy as np

def random_point_unit_ball(dim: int = 3) -> np.ndarray:
    vec = np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec

def random_so3_cayley() -> np.ndarray:
    a = np.random.randn(3)
    
    A = np.array([[ 0, -a[2], a[1]],
                  [ a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])

    # Find SO(3) using Cayley's parametrization: R = (I - A) @ inv(I + A)
    I = np.eye(3)
    R = np.linalg.inv(I + A) @ (I - A) 
    
    return R