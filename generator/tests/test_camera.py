import unittest
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import Camera, random_points, special_transform, so3_to_cayley, isSO3, normalize
from points import random_point_on_ball, random_point_inside_ball, random_so3_cayley

class CameraTest(unittest.TestCase):

    def test_isSO3_true(self):
        R = np.eye(3)
        self.assertTrue(isSO3(R))

    def test_isSO3_false(self):
        R = np.eye(3) * 2
        self.assertFalse(isSO3(R))

    def test_normalize(self):
        v = np.array([3, 0, 0])
        result = normalize(v)
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_special_transform(self):
        p1 = np.array([2, 3, 1])
        p2 = np.array([10, 4, 1.0])

        T = special_transform(p1, p2)

        self.assertEqual(T.shape, (3, 3))
        self.assertTrue(isSO3(T))

        # Check that T @ p1 looks like [0, 0, a] and T @ p2 looks like [0, b, c]
        Tp1 = T @ p1
        Tp2 = T @ p2
        self.assertAlmostEqual(Tp1[0], 0, places=6)
        self.assertAlmostEqual(Tp1[1], 0, places=6)
        self.assertAlmostEqual(Tp1[2], np.linalg.norm(p1), places=6)
        self.assertAlmostEqual(Tp2[0], 0, places=6)
        self.assertNotAlmostEqual(Tp2[1], 0, places=6)
        self.assertNotAlmostEqual(Tp2[2], 0, places=6)


    def test_so3_to_cayley_identity(self):
        R = np.eye(3)
        cayley = so3_to_cayley(R)
        expected = np.zeros(3)
        np.testing.assert_array_almost_equal(cayley, expected)

    def test_camera_projection(self):
        R = np.eye(3)
        t = np.array([0, 0, 0])
        cam = Camera(R, t)
        point = np.array([1, 2, 1])
        projected = cam.project(point)
        expected = np.array([1, 2, 1])
        np.testing.assert_array_almost_equal(projected, expected)

    def test_camera_invalid_rotation_shape(self):
        R = np.eye(2)
        t = np.zeros(3)
        with self.assertRaises(AssertionError):
            Camera(R, t)

    def test_camera_invalid_translation_shape(self):
        R = np.eye(3)
        t = np.zeros(2)
        with self.assertRaises(AssertionError):
            Camera(R, t)
