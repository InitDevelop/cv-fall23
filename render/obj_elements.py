from render.open_obj import read_file
import numpy as np

from constants.constants import *

vertices, lines, faces, face_normals = read_file("../objects/laptop.obj")

scene_points = vertices * scale
z_depth = 0#-np.min(scene_points[:, 2]) * depth_ratio
scene_points[:, 2] += z_depth

scene_faces = faces
scene_lines = lines
scene_normals = face_normals

face_colors = np.ones((scene_faces.shape[0], 3)) * 0.85
# np.random.random((scene_faces.shape[0], 3))
#

pov_pos_inch = np.array([20, -15, -20])
pov_pos = pov_pos_inch * ppi

inter_ratio = pov_pos[2] / (pov_pos[2] - scene_points[:, 2])
inter_ratio = np.stack((inter_ratio, inter_ratio, inter_ratio), axis=1)
scene_points_converted = pov_pos + inter_ratio * (scene_points - pov_pos)
scene_points_converted = scene_points_converted[:, 0:2] + np.array([screen_width / 2, screen_height / 2])

cam_pos = pov_pos
# cam_pos = np.array([0, 0, -200])
cam_dir = np.array([-pov_pos[0], -pov_pos[1], z_depth - pov_pos[2]])
cam_dir /= np.linalg.norm(cam_dir, ord=2)

point_depths = (scene_points - cam_pos) @ cam_dir / 1200
depth_nums = np.sum((scene_points[faces[:, 0]] - cam_pos) * scene_normals, axis=1)

# face_ind = np.arange(faces.shape[0])

culling = depth_nums < 0  # < 0
faces = faces[culling]

# face_colors[:] = 1

#ace_colors = env.lambertian(frame, scene_normals[culling]) * face_colors[culling]

