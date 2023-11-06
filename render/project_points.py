import numpy as np
import torch

from render.render_wireframe import *
from supports.logger import *


def default_projector(points, width, height, camera_angle, pose):
    # points : (N,3)
    points_homo = np.ones((points.shape[0], 4))
    points_homo[:, 0:3] = points

    f = width / (2 * np.tan(camera_angle / 360 * np.pi))

    K = np.array([
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ])

    proj = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    proj_points = K @ proj @ pose @ points_homo.T
    proj_points = proj_points[0:2] / proj_points[2]

    return proj_points.T  # (N,2)


def default_rays(width, height, camera_angle, pose):
    uv = np.indices((height, width)) - np.array([height / 2, width / 2]).reshape(2,1,1)

    f = width / (2 * np.tan(camera_angle / 360 * np.pi))

    uv = np.vstack([uv, f * np.ones((1,height, width)), np.ones((1,height, width))])
    uv = np.rollaxis(uv, 0, 2)

    rays = np.linalg.inv(pose) @ uv

    rays = np.rollaxis(rays, 1, 3)

    rays = rays[..., 0:3] / rays[..., 3:4]

    return rays


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    delay_logger = logger()
    delay_logger.set_log("delay : %s ms")

    cube_points = np.array(
        [(100, -200, -100), (-100, -200, -100), (-100, 0, -100), (100, 0, -100), (100, -200, 100), (-100, -200, 100),
         (-100, 0, 100), (100, 0, 100), ])

    cube_lines = np.array(
        [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])

    cube_points_2 = cube_points / 4 + np.array([0, -75, 0])
    cube_points_3 = cube_points / 2 - np.array([0, -50, 0])

    pose = np.eye(4, 4)
    pose[0:3, 3] = [0, 100, 800]
    projected_points = default_projector(cube_points, 1280, 720, 60, pose)

    screen = Screen()
    start = time.time()
    delay = 0
    count = 0

    while True:
        delay_start = time.time()
        count += 1
        screen.flush()
        dt = time.time() - start
        # pose[0:3,3] = [150 * np.cos(dt), 100 + 150 * np.sin(dt),800 + 400 * np.sin(1.7 * dt)]

        rotate_mat_y = np.array([
            [np.cos(2 * dt), 0, np.sin(2 * dt)],
            [0, 1, 0],
            [-np.sin(2 * dt), 0, np.cos(2 * dt)]
        ])

        rotate_mat_z = np.array([
            [np.cos(dt), -np.sin(dt), 0],
            [np.sin(dt), np.cos(dt), 0],
            [0, 0, 1]
        ])

        cube_points_rot = rotate_mat_y @ cube_points_2.T
        cube_points_rot_2 = rotate_mat_z @ cube_points_3.T

        projected_points = default_projector(cube_points, 1280, 720, 60, pose)
        projected_points_2 = default_projector(cube_points_rot.T, 1280, 720, 60, pose)
        projected_points_3 = default_projector(cube_points_rot_2.T + np.array([0, -100, 0]), 1280, 720, 60, pose)

        render_wireframe(screen, projected_points, cube_lines)
        render_wireframe(screen, projected_points_2, cube_lines)
        render_wireframe(screen, projected_points_3, cube_lines)

        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        delay += time.time() - delay_start
        if count >= 100:
            delay_logger.print(delay / count * 1000)
            delay = 0
            count = 0

        screen.fps.tick(screen.fps_aim)
