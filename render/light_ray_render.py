from render.open_obj import read_file
from render.project_points import *
from render.environment import *

flip_arr = np.array([[0, 1], [1, 0]])

# Global Parameters
scale = 150  # obj to pixel scale
depth_ratio = 1.8
depth_margin_ratio = 0.4    # smaller than 0.5
screen_height, screen_width = 720, 1280
camera_height, camera_width = 720, 1280


@jit(cache=True)
def get_depth_map(map, color_map, points, proj_points, faces, normals, face_colors, frame, env, z_depth, pov_pos):
    # points : N,3
    # projected_points : N,2
    # faces : F,3
    # cam_pos : 3,
    # cam_dir : 3,
    cam_pos = pov_pos
    cam_dir = np.array([-pov_pos[0], -pov_pos[1], -pov_pos[2] + z_depth])
    cam_dir /= np.linalg.norm(cam_dir, ord=2)

    point_depths = (points - cam_pos) @ cam_dir

    min_depth = np.min(point_depths) * (1 - depth_margin_ratio)
    depth_range = np.max(point_depths) * (1 + depth_margin_ratio) - min_depth

    point_depths = (point_depths - min_depth) / depth_range
    depth_nums = np.sum((points[faces[:, 0]] - cam_pos) * normals, axis=1)

    culling = depth_nums < 0
    faces = faces[culling]

    face_colors = env.lambertian(frame, normals[culling]) * face_colors[culling]

    proj_points = proj_points @ flip_arr
    # index_map = -np.ones((height, width))

    for i in range(faces.shape[0]):
        face = faces[i]
        face_color = face_colors[i]
        render_polygon(map, color_map, proj_points[face], point_depths[face], face_color)


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
            start = min(map.shape[1], max(math.floor(head_pos), 0))
            end = min(map.shape[1], max(math.ceil(tail_pos) + 1, 0))
            start_depth = (start - head_pos) * (tail_depth - head_depth) / (tail_pos - head_pos) + head_depth
            end_depth = (end - head_pos) * (tail_depth - head_depth) / (tail_pos - head_pos) + head_depth
        else:
            start = min(map.shape[1], max(math.floor(tail_pos), 0))
            end = min(map.shape[1], max(math.ceil(head_pos) + 1, 0))
            start_depth = (start - tail_pos) * (head_depth - tail_depth) / (head_pos - tail_pos) + tail_depth
            end_depth = (end - tail_pos) * (head_depth - tail_depth) / (head_pos - tail_pos) + tail_depth

        if start == end:
            continue
        # print(start_depth, end_depth, start, end)
        depths_line = np.arange(0, end - start) / (end - start) * (end_depth - start_depth) + start_depth
        color_map[int(x), start:end][depths_line < map[int(x), start:end, 0]] = color
        map[int(x), start:end, 0] = np.minimum(depths_line, map[int(x), start:end, 0])

    return map


@jit
def render_frame(frame, pov_pos, scene_points, scene_lines, scene_faces, scene_normals, face_colors,
                 start, delay, count, map, color_map, env, z_depth):
    inter_ratio = pov_pos[2] / (pov_pos[2] - scene_points[:, 2])
    inter_ratio = np.stack((inter_ratio, inter_ratio, inter_ratio), axis=1)
    scene_points_converted = pov_pos + inter_ratio * (scene_points - pov_pos)
    scene_points_converted = scene_points_converted[:, 0:2] + np.array([screen_width / 2, screen_height / 2])

    map *= 0
    map += 1

    color_map *= 0

    get_depth_map(map.numpy(), color_map.numpy(), scene_points, scene_points_converted, scene_faces,
                  scene_normals, face_colors, frame, env, z_depth, pov_pos)

    map -= 1
    map *= -1.5
    color_map *= map

    return color_map.numpy()


@jit
def render_scene(scene_points, scene_lines, scene_faces, scene_normals, face_colors, z_depth):
    env = Environment(32, 18, 120)

    start = time.time()
    delay = 0
    count = 0

    map = torch.ones((screen_height, screen_width, 1), device=device)
    color_map = torch.zeros((screen_height, screen_width, 3), device=device)

    capture_video(screen_width, screen_height, camera_width, camera_height, render_frame, True, scene_points,
                  scene_lines, scene_faces, scene_normals, face_colors,
                  start, delay, count, map, color_map, env, z_depth)


def initiate(path):
    delay_logger = logger()
    delay_logger.set_log("delay : %s ms")

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices, lines, faces, face_normals = read_file(path)

    scene_points = vertices * scale

    scene_faces = faces
    scene_lines = lines

    #scene_points[:, 2] += - np.min(scene_points[:, 2]) * depth_ratio

    max_z = np.max(scene_points[:, 2])
    min_z = np.min(scene_points[:, 2])

    z_depth = (max_z + min_z) / 2

    scene_normals = face_normals

    face_colors = np.ones((scene_faces.shape[0], 3))
    # face_colors = np.random.random((scene_faces.shape[0], 3))

    render_scene(scene_points, scene_lines, scene_faces, scene_normals, face_colors, z_depth)
