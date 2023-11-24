import numpy as np


def get_vertex_as_list(x_token, y_token, z_token):
    """
    :param x_token: string that contains the x value of the vertex
    :param y_token: string that contains the y value of the vertex
    :param z_token: string that contains the z value of the vertex
    :return: NumPy array of 3 float values of x, y, z
    """
    return np.array([float(x_token), float(y_token), float(z_token)], dtype=np.float32)


def get_triple_face(point_1, point_2, point_3):
    """
    The first, second, third point token of 'f' statement in the obj file
    :param point_1: First point token
    :param point_2: Second point token
    :param point_3: Third point token
    :return: tuple of 3 vertex indexes, list of 3 tuples for representing each line, array of 3 vertex normal indexes
    """
    p1 = int(point_1.split('/')[0]) - 1
    p2 = int(point_2.split('/')[0]) - 1
    p3 = int(point_3.split('/')[0]) - 1

    vn_idx_1 = int(point_1.split('/')[2]) - 1
    vn_idx_2 = int(point_2.split('/')[2]) - 1
    vn_idx_3 = int(point_3.split('/')[2]) - 1

    return ((p1, p2, p3), [(p1, p2), (p2, p3), (p3, p1)],
            np.array([vn_idx_1, vn_idx_2, vn_idx_3], dtype=int))


def get_quadruple_face(point_1, point_2, point_3, point_4):
    """
    The first, second, third, fourth point token of 'f' statement in the obj file
    :param point_1: First point token
    :param point_2: Second point token
    :param point_3: Third point token
    :param point_4: Fourth point token
    :return: 2 tuples of vertex indexes, list of 5 tuples for representing each line, 2 lists of 3 vertex normal indexes
    """
    p1 = int(point_1.split('/')[0]) - 1
    p2 = int(point_2.split('/')[0]) - 1
    p3 = int(point_3.split('/')[0]) - 1
    p4 = int(point_4.split('/')[0]) - 1

    vn_idx_1 = int(point_1.split('/')[2]) - 1
    vn_idx_2 = int(point_2.split('/')[2]) - 1
    vn_idx_3 = int(point_3.split('/')[2]) - 1
    vn_idx_4 = int(point_4.split('/')[2]) - 1

    return ((p1, p2, p4), (p2, p3, p4), [(p1, p2), (p2, p3), (p3, p4), (p4, p1), (p2, p4)],
            np.array([vn_idx_1, vn_idx_2, vn_idx_4], dtype=int),
            np.array([vn_idx_2, vn_idx_3, vn_idx_4], dtype=int))


def read_file(path):
    """
    :param path: Path of the obj file
    :return: vertices, lines, faces, face_normals
    """
    vertices = []
    vertex_normals = []
    lines = []
    faces = []
    face_normals = []

    with open(path, 'r') as file:
        string_list = file.readlines()

        finished = False

        for line in string_list:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == 'v':
                vertices.append(get_vertex_as_list(tokens[1], tokens[2], tokens[3]))
            elif tokens[0] == 'vn':
                vertex_normals.append(get_vertex_as_list(tokens[1], tokens[2], tokens[3]))
            elif tokens[0] == 'f':
                # If token[0] is 'f', then this means that 'v' and 'vn' tokens are all read
                if not finished:
                    vertices = np.array(vertices)
                    vertex_normals = np.array(vertex_normals)
                    finished = True

                if len(tokens) == 4:
                    f_vert, t_list, vn = get_triple_face(tokens[1], tokens[2], tokens[3])
                    lines.extend(t_list)
                    faces.append(f_vert)
                    vn_mat = vertex_normals[vn]
                    face_normals.append([(vn_mat[0] + vn_mat[1] + vn_mat[2]) / 3])

                elif len(tokens) == 5:
                    f_vert_1, f_vert_2, t_list, vn_1, vn_2 = get_quadruple_face(
                        tokens[1], tokens[2], tokens[3], tokens[4]
                    )
                    lines.extend(t_list)
                    faces.append(f_vert_1)
                    faces.append(f_vert_2)
                    vn_mat_1 = vertex_normals[vn_1]
                    face_normals.append([(vn_mat_1[0] + vn_mat_1[1] + vn_mat_1[2]) / 3])
                    vn_mat_2 = vertex_normals[vn_2]
                    face_normals.append([(vn_mat_2[0] + vn_mat_2[1] + vn_mat_2[2]) / 3])

    face_normals = np.array(face_normals, dtype=np.float32)
    lines = np.array(lines)
    faces = np.array(faces)

    return vertices, lines, faces, face_normals


# DEBUG TEST CODE
if __name__ == "__main__":
    vertices, lines, faces, face_normals = read_file("E:\\polyfile.obj")
    print(vertices.shape)
    print(lines.shape)
    print(faces.shape)
    print(face_normals.shape)

