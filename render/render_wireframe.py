from render.screen import *


def render_wireframe(screen, points, lines, width=1):
    for srt, end in lines:
        pygame.draw.line(screen.display, screen.BLACK, points[srt], points[end], width=width)
