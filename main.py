import numpy as np
import torch

from render.light_ray_render import render_scene
from render.open_obj import read_file
from supports.logger import *

if __name__ == "__main__":
    delay_logger = logger()
    delay_logger.set_log("delay : %s ms")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices, lines, faces, face_normals = read_file("./objects/laptop.obj")
    # laptop.obj source : https://free3d.com/3d-model/notebook-low-poly-version-57341.html

    scene_points = vertices
    scene_faces = faces
    scene_lines = lines
    scene_normals = face_normals

    face_colors = np.ones((scene_faces.shape[0], 3)) * 0.85
    # np.random.random((scene_faces.shape[0], 3))
    #

    pose = np.eye(4, 4)
    pose[0:3, 3] = [0, 0, 12]
    camera_pos = np.array([0, 0, -12])
    camera_direction = np.array([0, 0, 1])

    render_scene(scene_points, scene_lines, scene_faces, scene_normals, face_colors, pose, camera_pos, camera_direction)