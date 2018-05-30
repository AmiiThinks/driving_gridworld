from pycolab import ascii_art
from pycolab import rendering
from pycolab.prefab_parts import drapes as prefab_drapes

from .road import road_art, bump_indices, pedestrian_indices, Road
from .obstacles import Bump, Pedestrian
from .car import Car
from .drapes import DitchDrape
from .sprites import BumpSprite, PedestrianSprite, car_sprite_class


def game_board(num_rows):
    assert num_rows > 1
    return ['|    |'] * num_rows


class RoadPycolabEnv(object):
    def __init__(self,
                 num_rows=5,
                 num_bumps=3,
                 num_pedestrians=3,
                 speed=1,
                 speed_limit=3):
        self._speed = speed
        self._speed_limit = speed_limit

        self._num_rows = num_rows
        self._num_bumps = num_bumps
        self._num_pedestrians = num_pedestrians

        ra = road_art(num_rows, num_bumps, num_pedestrians)
        gb = game_board(num_rows)
        bi = bump_indices(num_bumps)
        bump_repaint_mapping = {c: 'b' for c in bi}
        pi = pedestrian_indices(num_pedestrians, num_bumps)
        pedestrian_repaint_mapping = {c: 'p' for c in pi}

        scrolly_info = prefab_drapes.Scrolly.PatternInfo(
            ra, gb, board_northwest_corner_mark='+', what_lies_beneath='|')

        sprites = {
            c: ascii_art.Partial(BumpSprite, scrolly_info.virtual_position(c))
            for c in bi if c in ''.join(ra)
        }
        sprites['C'] = ascii_art.Partial(
            car_sprite_class(speed=speed, speed_limit=speed_limit),
            scrolly_info.virtual_position('C'))
        for c in pi:
            if c in ''.join(ra):
                sprites[c] = ascii_art.Partial(
                    PedestrianSprite, scrolly_info.virtual_position(c))
        self._game = ascii_art.ascii_art_to_game(
            gb,
            what_lies_beneath=' ',
            sprites=sprites,
            drapes={
                'd': ascii_art.Partial(DitchDrape, **scrolly_info.kwargs('d'))
            },
            update_schedule=[(['d'] + list(bump_repaint_mapping.keys()) +
                              list(pedestrian_repaint_mapping.keys())), ['C']],
            z_order='d' + bi + pi + 'C')

        repaint_mapping = {}
        for k, v in bump_repaint_mapping.items():
            repaint_mapping[k] = v
        for k, v in pedestrian_repaint_mapping.items():
            repaint_mapping[k] = v
        self._repainter = rendering.ObservationCharacterRepainter(
            repaint_mapping)

    def its_showtime(self):
        observation, reward, discount = self._game.its_showtime()
        observation = self._repainter(observation)
        return (observation, self._speed), reward, discount

    def play(self, a):
        if a == 0:
            self._speed = min(self._speed + 1, self._speed_limit)
        elif a == 1:
            self._speed = max(self._speed - 1, 1)
        observation, reward, discount = self._game.play(a)
        observation = self._repainter(observation)
        return (observation, self._speed), reward, discount

    def with_walls_removed(self, board):
        return board[:, 1:-1]

    def observation_to_key(self, o):
        pycolab_observation, speed = o
        board_array = self.with_walls_removed(pycolab_observation.board)
        ascii_board_rows = [
            board_array[i].tostring().decode('ascii')
            for i in range(len(board_array))
        ]
        return ('\n'.join(ascii_board_rows), speed)

    def to_road(self):
        car = Car(self._num_rows - 1, 2, self._speed)
        initial_bumps = [Bump(-1, -1) for _ in range(self._num_bumps)]
        initial_pedestrians = [
            Pedestrian(-1, -1) for _ in range(self._num_pedestrians)
        ]
        return Road(self._num_rows, car, initial_bumps + initial_pedestrians,
                    self._speed_limit)
