import numpy as np


class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


def vector_to_array(x=0, y=0, z=0):
    return np.array([x, y, z]).T


def to_homogeneous(np_arr):
    return np.array([np_arr[0, 0], np_arr[1, 0], np_arr[2, 0], 1]).T

