import cv2
from cv_functions.capture_video import *


def downscale(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def show_downscale(frame, width, height):
    return cv2.resize(downscale(frame, width, height), (1280, 720), interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    capture_video(1280, 720, show_downscale, True, 32, 18)
