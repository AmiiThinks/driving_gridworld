from .road import Road
from .obstacles import Bump, Pedestrian
from .car import Car
from .actions import UP, DOWN, QUIT


def game_board(num_rows):
    assert num_rows > 1
    return ['|    |'] * num_rows


class DrivingGridworld(object):
    def __init__(self, num_rows=5, num_bumps=3, num_pedestrians=3, speed=1):
        self.the_plot = {}
        self.game_over = False
        self._initial_speed = speed
        self._speed = speed

        self._num_rows = num_rows
        self._num_bumps = num_bumps
        self._num_pedestrians = num_pedestrians

    def _speed_limit(self):
        return self._num_rows + 1

    def its_showtime(self):
        # TODO:
        pass

    def play(self, a):
        if a == QUIT:
            self.game_over = True
        elif a == UP:
            self._speed = min(self._speed + 1, self._speed_limit())
        elif a == DOWN:
            self._speed = max(self._speed - 1, 1)
        pass

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

    def to_road(self, o=None):
        if o is None:
            car = Car(self._num_rows - 1, 2, self._initial_speed)
            initial_bumps = [Bump(-1, -1) for _ in range(self._num_bumps)]
            initial_pedestrians = [
                Pedestrian(-1, -1) for _ in range(self._num_pedestrians)
            ]
            return Road(self._num_rows,
                        car, initial_bumps + initial_pedestrians,
                        self._speed_limit())
        else:
            pass
