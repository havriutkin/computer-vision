""" Methods for generating points on a unit Ball and elements of SO(3) group """
import numpy as np

def random_point_on_ball(dim: int = 3, radius: float = 1) -> np.ndarray:
    """ Generate a random point on the surface of a unit ball in n-dimensions with given radius """
    vec = np.random.randn(dim)
    vec /= np.linalg.norm(vec) 
    vec *= radius 

    return vec

def random_point_inside_ball(dim: int = 3, radius: float = 1, threshold: float = 0) -> np.ndarray:
    """ Generate a random point inside a unit ball in n-dimensions with given radius """
    vec = np.random.randn(dim)
    vec /= np.linalg.norm(vec) 
    vec *= np.random.uniform(0, radius - threshold) 

    return vec

def random_so3_cayley() -> np.ndarray:
    """ Generate a random rotation matrix in SO(3) using Cayley's parametrization """
    a = random_point_on_ball(dim=3, radius=1)  

    A = np.array([[ 0, -a[2], a[1]],
                  [ a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])

    # Find SO(3) using Cayley's parametrization: R = (I - A) @ inv(I + A)
    I = np.eye(3)
    R = np.linalg.inv(I + A) @ (I - A) 
    
    return R