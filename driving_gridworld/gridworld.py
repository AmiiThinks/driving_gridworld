from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import QUIT, NO_OP, LIST_CONTROLS
from collections import namedtuple


def simple_road_factory(headlight_range=5,
                        num_bumps=3,
                        num_pedestrians=3,
                        speed=1,
                        discount=0.99,
                        bump_appearance_prob=0.2,
                        pedestrian_appearance_prob=0.2,
                        car_col=2):
    car = Car(car_col, speed)
    initial_bumps = [
        Bump(-1, -1, prob_of_appearing=bump_appearance_prob)
        for _ in range(num_bumps)
    ]
    initial_pedestrians = [
        Pedestrian(-1, -1, prob_of_appearing=pedestrian_appearance_prob)
        for _ in range(num_pedestrians)
    ]
    return Road(headlight_range, car, initial_bumps + initial_pedestrians)


class DrivingGridworld(object):
    Backdrop = namedtuple('Backdrop', ['palette'])

    @classmethod
    def legacy_constructor(cls,
                           headlight_range=5,
                           num_bumps=3,
                           num_pedestrians=3,
                           speed=1,
                           discount=0.99,
                           bump_appearance_prob=0.2,
                           pedestrian_appearance_prob=0.2,
                           car_col=2):
        return cls(
            (
                lambda: simple_road_factory(
                    headlight_range=headlight_range,
                    num_bumps=num_bumps,
                    num_pedestrians=num_pedestrians,
                    speed=speed,
                    discount=discount,
                    bump_appearance_prob=bump_appearance_prob,
                    pedestrian_appearance_prob=pedestrian_appearance_prob,
                    car_col=car_col)
            ),
            discount=discount
        )

    def __init__(self, road_factory=simple_road_factory, discount=0.99):
        self._road_factory = road_factory
        self._discount = discount
        self.reset()

    def reset(self):
        self.game_over = False
        self.road = self._road_factory()

        # For compatibility with pycolab croppers.
        self.the_plot = {}
        self.things = []
        self.backdrop = self.Backdrop([])

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
        self.road = self.road.sample_transition(a)
        discount = self._discount
        if self.road.has_crashed():
            self.game_over = True
            discount = 0.0
        return self.road, 0, discount

    def observation_to_key(self, o):
        ascii_board_rows = [
            o.board[i].tostring().decode('ascii') for i in range(len(o.board))
        ]
        return ''.join(ascii_board_rows)


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
