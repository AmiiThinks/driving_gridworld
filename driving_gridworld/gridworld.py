from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import QUIT, NO_OP, LIST_CONTROLS
from collections import namedtuple


class DrivingGridworld(object):
    Backdrop = namedtuple('Backdrop', ['palette'])

    def __init__(self,
                 headlight_range=5,
                 num_bumps=3,
                 num_pedestrians=3,
                 speed=1,
                 discount=0.99,
                 bump_appearance_prob=0.2,
                 pedestrian_appearance_prob=0.2,
                 car_col=2):
        self.the_plot = {}
        self.game_over = False
        self._initial_speed = speed
        self._speed = speed
        self._discount = discount
        self._car_col = car_col

        self._headlight_range = headlight_range
        self._num_bumps = num_bumps
        self._num_pedestrians = num_pedestrians
        self._bump_appearance_prob = bump_appearance_prob
        self._pedestrian_appearance_prob = pedestrian_appearance_prob

        self.reset()

        # For compatibility with pycolab croppers.
        self.things = []
        self.backdrop = self.Backdrop([])

    def reset(self):
        self.game_over = False
        self.car = Car(self._car_col, self._initial_speed)
        initial_bumps = [
            Bump(-1, -1, prob_of_appearing=self._bump_appearance_prob)
            for _ in range(self._num_bumps)
        ]
        initial_pedestrians = [
            Pedestrian(
                -1, -1, prob_of_appearing=self._pedestrian_appearance_prob)
            for _ in range(self._num_pedestrians)
        ]
        self.road = Road(self._headlight_range, self.car,
                         initial_bumps + initial_pedestrians)
        return self

    def its_showtime(self):
        self.reset()
        return self.road.observation(), 0.0, self._discount

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
        discount = self._discount
        if self.road.has_crashed():
            self.game_over = True
            reward += (self.road.reward_for_being_in_transit /
                       (1.0 - self._discount))
            discount = 0.0
        return self.road, reward, discount

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
