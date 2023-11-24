import cv2
import time
from supports.logger import *


def capture_video(width, height, function, show_log=True, *args):
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps_logger = logger()
    fps_logger.set_log("current fps : %d, delay : %d ms")

    start = time.time()
    count = 0
    delay = 0
    delay_start = 0

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
        frame = function(frame, *args)
        cv2.imshow("ImmerVision", frame)
        if show_log:
            count += 1
            delay += time.time() - delay_start

    capture.release()
    cv2.destroyAllWindows()


def null_function(x):
    return x


def canny_function(img):
    return cv2.Canny(img, 100, 200)


if __name__ == "__main__":
    capture_video(1280, 720, canny_function)
