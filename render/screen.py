import sys
import pygame
from pygame.locals import *
import math
import time


class Screen:
    def __init__(self, width=1280, height=720, fps=30, caption="example screen"):
        pygame.init()
        self.fps = pygame.time.Clock()
        self.fps_aim = fps

        self.display = pygame.display.set_mode((width, height))
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        self.display.fill(self.WHITE)  # fill bg to white
        pygame.display.set_caption(caption)  # screen window caption

    def draw_lines(self, lines):
        for line in lines:
            pygame.draw.line(self.display, self.BLACK, line[0], line[1])

    def draw_dlines(self, dpoints):
        last_pos = dpoints[0]
        for point in dpoints[1:]:
            now_pos = (last_pos[0] + point[0], last_pos[1] + point[1])
            pygame.draw.line(self.display, self.BLACK, last_pos, now_pos, width=5)
            last_pos = now_pos

    def flush(self):
        self.display.fill(self.WHITE)


if __name__ == "__main__":
    screen = Screen()
    screen.draw_lines([((100, 100), (500, 300)), ((500, 300), (900, 200))])
    start = time.time()
    while True:
        screen.flush()
        dt = time.time() - start
        screen.draw_dlines([
            (640, 360),
            (200 * math.cos(dt), 200 * math.sin(dt)),
            (100 * math.cos(2 * dt), 100 * math.sin(2 * dt)),
            (50 * math.cos(4 * dt), 50 * math.sin(4 * dt)),
            (25 * math.cos(8 * dt), 25 * math.sin(8 * dt)),
        ])
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        screen.fps.tick(screen.fps_aim)
