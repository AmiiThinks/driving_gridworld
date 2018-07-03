import curses
from pycolab import human_ui
from pycolab.rendering import ObservationToArray
import numpy as np

from .actions import UP, DOWN, LEFT, RIGHT, NO_OP, QUIT, LIST_CONTROLS
from .gridworld import RecordingDrivingGridworld


def color256_to_1000(c):
    return int(c / 255.0 * 999)


COLOUR_FG = {
    ' ': (color256_to_1000(183), color256_to_1000(177), color256_to_1000(174)),
    '|': (color256_to_1000(67), color256_to_1000(70), color256_to_1000(75)),
    'd': (color256_to_1000(87), color256_to_1000(59), color256_to_1000(12)),
    'C': (0, 999, 999),
    'b': (0, 0, 0),
    'p': (987, 623, 145)
}
COLOUR_BG = {}
obs_to_rgb = ObservationToArray(COLOUR_FG)


def observation_to_img(o):
    return np.swapaxes(obs_to_rgb(o) / 1000.0, 1, 2).T


class UiRecordingDrivingGridworld(RecordingDrivingGridworld):
    def ui_play(self):
        ui = human_ui.CursesUi(
            keys_to_actions={
                curses.KEY_UP: UP,
                curses.KEY_DOWN: DOWN,
                curses.KEY_LEFT: LEFT,
                curses.KEY_RIGHT: RIGHT,
                -1: NO_OP,
                'q': QUIT,
                'Q': QUIT,
                'l': LIST_CONTROLS,
                'L': LIST_CONTROLS
            },
            delay=1000,
            colour_fg=COLOUR_FG,
            colour_bg=COLOUR_BG)
        return ui.play(self)
