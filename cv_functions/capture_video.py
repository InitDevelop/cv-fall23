import cv2
import time

import numpy as np

from built_in.face_detection_mediapipe import DrawMesh
from supports.logger import *

depth_in_pixels = 6000
eye_separation_inches = 3.5
fov_camera = 100
ppi = 150


def capture_video(width, height, camera_width, camera_height, function, show_log=True, *args):
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps_logger = logger()
    fps_logger.set_log("current fps : %d, delay : %d ms")

    start = time.time()
    count = 0
    delay = 0
    delay_start = 0

    depth_const = ppi * camera_width * eye_separation_inches / (2 * np.tan(fov_camera * np.pi / 360))

    mesh = DrawMesh()

    while cv2.waitKey(1) < 0:
        if show_log:
            if time.time() - start >= 1:
                fps_logger.print(count / (time.time() - start), delay * 1000 / count)
                start = time.time()
                count = 0
                delay = 0
        ret, frame = capture.read()
        if show_log:
            delay_start = time.time()
        frame = cv2.flip(frame, 1)
        face_frame = np.copy(frame)
        face_frame = mesh.face_mesh(face_frame)

        eye_pos = mesh.get_eye_position(face_frame)

        depth = 1500    # depth_const / np.abs(eye_pos[2])

        pov_pos = np.array(
            [eye_pos[0] - camera_width / 2,
             eye_pos[1] - camera_height / 2,
             -depth])

        frame = function(face_frame, pov_pos, *args)
        cv2.imshow("ImmerVision", frame)
        if show_log:
            count += 1
            delay += time.time() - delay_start

    capture.release()
    cv2.destroyAllWindows()


def null_function(x, y):
    return x


def canny_function(img):
    return cv2.Canny(img, 100, 200)


# if __name__ == "__main__":
#     capture_video(1280, 720, null_function)

