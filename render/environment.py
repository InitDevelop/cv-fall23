import numpy as np
from cv_functions.downscale_video import *
from numba import jit


class Environment:
    def __init__(self, width, height, camera_angle):
        self.width = width
        self.height = height
        self.f = width / 2 / np.tan(camera_angle / 2)
        xy = np.indices((height, width))
        xyz = np.vstack([xy[::-1], -self.f * np.ones((1,height, width))])
        xyz = np.rollaxis(xyz, 0, 3) - np.array([width/2, height/2, 0])
        xyz[...,0] *= -1

        norm = np.linalg.norm(xyz, axis=2, keepdims=True)
        self.env = (xyz / norm).reshape(-1, 3)


    @jit(cache=True)
    def lambertian(self, img, normals):
        # normals : (F, 3)
        # img : (1280, 720)
        # env : (32, 18, 3)

        d_img = downscale(img, self.width, self.height).reshape(-1, 3) # 32*18, 3
        activation = self.env @ normals.T  # 32*18, F

        colors = d_img.T @ activation / (self.width * self.height * 255)
        colors = 2 * colors - np.power(colors, 2)

        return colors.T
