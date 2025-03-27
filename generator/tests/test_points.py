import unittest
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from points import random_point_on_ball, random_point_inside_ball, random_so3_cayley

class PointGenerationTest(unittest.TestCase):
    def test_random_point_on_ball(self):
        for dim in [2, 3, 4, 5]:
            for radius in [1, 2, 3, 4.5]:
                point = random_point_on_ball(dim=dim, radius=radius)

                self.assertEqual(point.shape, (dim,))
                self.assertAlmostEqual(np.linalg.norm(point), radius, places=6)

    def test_random_point_inside_ball(self):
        for dim in [2, 3, 4, 5]:
            for radius in [1, 2, 3, 4.5]:
                point = random_point_inside_ball(dim=dim, radius=radius)

                self.assertEqual(point.shape, (dim,))
                self.assertLessEqual(np.linalg.norm(point), radius)
                self.assertGreaterEqual(np.linalg.norm(point), 0)

    def test_random_point_inside_ball_with_threshold(self):
        for dim in [2, 3, 4, 5]:
            for radius in [1, 2, 3, 4.5]:
                for threshold in [0, 0.1, 0.5]:
                    point = random_point_inside_ball(dim=dim, radius=radius, threshold=threshold)

                    self.assertEqual(point.shape, (dim,))
                    self.assertLessEqual(np.linalg.norm(point), radius - threshold)
                    self.assertGreaterEqual(np.linalg.norm(point), 0)

    def test_random_so3_cayley(self):
        R = random_so3_cayley()
        
        # Check if R is a valid rotation matrix in SO(3)
        self.assertEqual(R.shape, (3, 3))
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6))
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)