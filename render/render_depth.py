import numpy as np
import sys

import pygame.mouse
from PIL import Image
import time
from numba import jit
import math
import torch
from render.project_points import *
from multiprocessing import Pool

flip_arr = np.array([[0, 1], [1, 0]])


@jit(parallel=True)
def get_depth_map(map, color_map, points, proj_points, faces, normals, face_colors, cam_pos, cam_dir):
    # points : N,3
    # projected_points : N,2
    # faces : F,3
    # cam_pos : 3,
    # cam_dir : 3,

    point_depths = (points - cam_pos) @ cam_dir / 1200
    depth_nums = np.sum((points[faces[:, 0]] - cam_pos) * normals, axis=1)

    # face_ind = np.arange(faces.shape[0])

    culling = depth_nums < 0
    faces = faces[culling]
    face_colors = face_colors[culling]

    proj_points = proj_points @ flip_arr
    # index_map = -np.ones((height, width))

    for i in range(faces.shape[0]):
        face = faces[i]
        face_color = face_colors[i]

        render_polygon(map, color_map, proj_points[face], point_depths[face], face_color)
        # render_triangle(depth_map, proj_points[face])


@jit(cache=True)
def render_polygon(map, color_map, v, depths, color):
    ind = np.argsort(v[:, 0])
    fv = v[ind]
    fd = depths[ind]
    N = len(v)

    # print(ind, fv)

    min_x = fv[0, 0]
    max_x = fv[-1, 0]

    head = ind[0] - 1
    tail = (ind[0] + 1) % N

    head_pos = fv[0, 1]
    tail_pos = fv[0, 1]

    head_depth = fd[0]
    tail_depth = fd[0]

    head_ramp = 0
    tail_ramp = 0
    head_dramp = 0
    tail_dramp = 0

    last_head_x = np.floor(min_x)
    last_tail_x = np.floor(min_x)

    for x in np.arange(max(0, np.ceil(min_x)), min(map.shape[0], np.ceil(max_x))):
        while True:
            update = False
            if v[(head + 1) % N, 0] < x:
                head = (head + 1) % N

                head_pos += head_ramp * (v[head, 0] - last_head_x)
                head_depth += head_dramp * (v[head, 0] - last_head_x)

                last_head_x = v[head, 0]

                head_ramp = v[(head + 1) % N] - v[head]
                head_dramp = depths[(head + 1) % N] - depths[head]

                if head_ramp[0]:
                    head_dramp = head_dramp / head_ramp[0]
                    head_ramp = head_ramp[1] / head_ramp[0]

                else:
                    head_pos += head_ramp[1]
                    head_depth += head_dramp
                    head_ramp = 0
                    head_dramp = 0

                update = True

            if v[tail - 1, 0] < x:
                tail -= 1

                tail_pos += tail_ramp * (v[tail, 0] - last_tail_x)
                tail_depth += tail_dramp * (v[tail, 0] - last_tail_x)

                last_tail_x = v[tail, 0]

                tail_ramp = v[tail - 1] - v[tail]
                tail_dramp = depths[tail - 1] - depths[tail]

                if tail_ramp[0]:
                    tail_dramp = tail_dramp / tail_ramp[0]
                    tail_ramp = tail_ramp[1] / tail_ramp[0]
                else:
                    tail_pos += tail_ramp[1]
                    tail_depth += tail_dramp
                    tail_ramp = 0
                    tail_dramp = 0

                update = True

            if head == ind[-1] or tail == ind[-1] - N:
                return

            if not update:
                head_pos += head_ramp * (x - last_head_x)
                tail_pos += tail_ramp * (x - last_tail_x)

                head_depth += head_dramp * (x - last_head_x)
                tail_depth += tail_dramp * (x - last_tail_x)

                last_head_x = x
                last_tail_x = x
                break

        # print(x,head_pos,tail_pos, head,tail)

        if head_pos < tail_pos:
            start = min(map.shape[1],max(math.floor(head_pos), 0))
            end = min(map.shape[1],max(math.ceil(tail_pos) + 1, 0))
            start_depth = (start - head_pos) * (tail_depth - head_depth) / (tail_pos - head_pos) + head_depth
            end_depth = (end - head_pos) * (tail_depth - head_depth) / (tail_pos - head_pos) + head_depth
        else:
            start = min(map.shape[1],max(math.floor(tail_pos), 0))
            end = min(map.shape[1],max(math.ceil(head_pos) + 1, 0))
            start_depth = (start - tail_pos) * (head_depth - tail_depth) / (head_pos - tail_pos) + tail_depth
            end_depth = (end - tail_pos) * (head_depth - tail_depth) / (head_pos - tail_pos) + tail_depth

        if start == end: continue
        # print(start_depth, end_depth, start, end)
        depths_line = np.arange(0, end - start) / (end - start) * (end_depth - start_depth) + start_depth
        filter = depths_line < map[int(x), start:end]
        map[int(x), start:end] = np.minimum(depths_line, map[int(x), start:end])
        color_map[start:end, int(x)][filter] = color

    return map


@jit(parallel=True)
def render_scene(scene_points, scene_lines, scene_faces, scene_normals, face_colors, pose, camera_pos, camera_direction):
    screen = Screen()
    start = time.time()
    delay = 0
    count = 0

    img = torch.zeros((1280, 720, 3), device=device)
    map = torch.ones((720, 1280), device=device)
    color_map = torch.zeros((1280,720,3), device=device)

    while True:
        delay_start = time.time()
        pos = pygame.mouse.get_pos()
        dt = (time.time() - start) * 0.8

        ry = dt * 1.7#-(pos[0] - 640) / 400
        rx = dt * 1.3#(pos[1] - 360) / 400

        rotate_mat_z = np.array([
            [np.cos(dt), -np.sin(dt), 0],
            [np.sin(dt), np.cos(dt), 0],
            [0, 0, 1]
        ])

        rotate_mat_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        rotate_mat_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        scene_points_rot = rotate_mat_x @ rotate_mat_y @ rotate_mat_z @ scene_points.T
        scene_normals_rot = rotate_mat_x @ rotate_mat_y @ rotate_mat_z @ scene_normals.T

        projected_points = default_projector(scene_points_rot.T, 1280, 720, 60, pose)

        map *= 0
        map += 1

        color_map *= 0

        get_depth_map(map.numpy(), color_map.numpy(), scene_points_rot.T, projected_points, scene_faces, scene_normals_rot.T, face_colors,
                      camera_pos,
                      camera_direction)

        map -= 1
        map *= -1.5

        color_map *= map.T.reshape(1280,720,1)

        # img[..., 0] = map.T
        # img[..., 1] = map.T
        # img[..., 2] = map.T

        # surf = pygame.surfarray.make_surface(map.T)
        # screen.display.blit(surf, (0, 0))
        pygame.surfarray.blit_array(screen.display, color_map.numpy())

        # render_wireframe(screen, projected_points, scene_lines)

        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        delay += time.time() - delay_start
        count += 1

        if count >= 100:
            delay_logger.print(delay / count * 1000)
            delay = 0
            count = 0

        screen.fps.tick(screen.fps_aim)


if __name__ == "__main__":
    delay_logger = logger()
    delay_logger.set_log("delay : %s ms")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cube_count = 27

    cube_points = np.array(
        [(100, -100, -100), (-100, -100, -100), (-100, 100, -100), (100, 100, -100), (100, -100, 100),
         (-100, -100, 100),
         (-100, 100, 100), (100, 100, 100), ]) / 2

    scene_points = np.vstack([
        cube_points,
        cube_points + np.array([150, 0, 0]),
        cube_points + np.array([-150, 0, 0]),
        cube_points + np.array([0, 0, 150]),
        cube_points + np.array([0, 0, -150]),
        cube_points + np.array([0, 150, 0]),
        cube_points + np.array([0, -150, 0]),
        cube_points + np.array([150, 0, 150]),
        cube_points + np.array([-150, 0, 150]),
        cube_points + np.array([-150, 0, -150]),
        cube_points + np.array([150, 0, -150]),
        cube_points + np.array([150, 150, 0]),
        cube_points + np.array([-150, 150, 0]),
        cube_points + np.array([0, 150, 150]),
        cube_points + np.array([0, 150, -150]),
        cube_points + np.array([150, 150, 150]),
        cube_points + np.array([-150, 150, 150]),
        cube_points + np.array([-150, 150, -150]),
        cube_points + np.array([150, 150, -150]),
        cube_points + np.array([150, -150, 0]),
        cube_points + np.array([-150, -150, 0]),
        cube_points + np.array([0, -150, 150]),
        cube_points + np.array([0, -150, -150]),
        cube_points + np.array([150, -150, 150]),
        cube_points + np.array([-150, -150, 150]),
        cube_points + np.array([-150, -150, -150]),
        cube_points + np.array([150, -150, -150])
    ])

    cube_lines = np.array(
        [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])

    scene_lines = np.vstack([cube_lines + 8 * i for i in range(cube_count)])

    cube_faces = np.array([
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7)
    ])

    scene_faces = np.vstack([cube_faces + 8 * i for i in range(cube_count)])

    face_colors = np.random.randint(0,255,(scene_faces.shape[0], 3))

    cube_normals = np.array([
        [0, 0, -1],
        [0, 0, 1],
        [0, -1, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [1, 0, 0]
    ])

    scene_normals = np.vstack([cube_normals for i in range(cube_count)])

    pose = np.eye(4, 4)
    pose[0:3, 3] = [0, 0, 800]
    camera_pos = np.array([0, 0, -800])
    camera_direction = np.array([0, 0, 1])
    '''
    start=time.time()
    rays = default_rays(1280,720,60,pose)
    print("rays:", time.time() - start)
    '''

    render_scene(scene_points, scene_lines, scene_faces, scene_normals, face_colors, pose, camera_pos, camera_direction)
