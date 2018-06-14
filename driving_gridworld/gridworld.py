from .road import Road
from .obstacles import Bump, Pedestrian
from .car import Car
from .actions import QUIT, NO_OP, LIST_CONTROLS
from collections import namedtuple


class DrivingGridworld(object):
    Backdrop = namedtuple('Backdrop', ['palette'])

    def __init__(self,
                 headlight_range=5,
                 num_bumps=3,
                 num_pedestrians=3,
                 speed=1,
                 discount=0.99):
        self.the_plot = {}
        self.game_over = False
        self._initial_speed = speed
        self._speed = speed
        self._discount = discount

        self._headlight_range = headlight_range
        self._num_bumps = num_bumps
        self._num_pedestrians = num_pedestrians

        self.car = Car(2, self._initial_speed)
        initial_bumps = [Bump(-1, -1) for _ in range(self._num_bumps)]
        initial_pedestrians = [
            Pedestrian(-1, -1) for _ in range(self._num_pedestrians)
        ]
        self.road = Road(self._headlight_range, self.car,
                         initial_bumps + initial_pedestrians)

        # For compatibility with pycolab croppers.
        self.things = []
        self.backdrop = self.Backdrop([])

    def its_showtime(self):
        return self.road.observation(), 0.0, 0.0

    def play(self, a):
        if a == QUIT:
            self.game_over = True
            return self.play(NO_OP)
        elif a == LIST_CONTROLS:
            # TODO: Show controls
            return self.play(NO_OP)
        else:
            road, reward, discount = self.fast_play(a)
            return road.observation(), reward, discount

    def fast_play(self, a):
        self.road, reward = self.road.sample_transition(a)
        return self.road, reward, self._discount

    def with_walls_removed(self, board):
        return board[:, 1:-1]

    def observation_to_key(self, o):
        board_array = self.with_walls_removed(o.board)
        ascii_board_rows = [
            board_array[i].tostring().decode('ascii')
            for i in range(len(board_array))
        ]
        return ('\n'.join(ascii_board_rows), self.speed())

    def speed(self):
        return self.car.speed


class RecordingDrivingGridworld(DrivingGridworld):
    def __init__(self, *args, **kwargs):
        self._recorded = []
        super().__init__(*args, **kwargs)

    def recorded(self):
        return self._recorded

    def its_showtime(self):
        observation, reward, discount = super().its_showtime()
        self._recorded = [(self.road.copy(), reward, discount)]
        return observation, reward, discount

    def play(self, action):
        observation, reward, discount = super().play(action)
        self._recorded.append((self.road.copy(), reward, discount, action))
        return observation, reward, discount
