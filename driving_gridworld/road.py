from itertools import product, permutations, combinations
import numpy as np
from pycolab.rendering import Observation


def _byte(c, encoding='ascii'):
    return bytes(c, encoding)[0]


def permutation_combination_pairs(a, b, n):
    return product(permutations(a, n), combinations(b, n))


class Road(object):
    # Drivable road definitions
    _paved_lanes = np.array([1, 2])
    _ditch_lanes = np.array([0, 3])
    _num_paved_lanes = len(_paved_lanes)
    _num_ditch_lanes = len(_ditch_lanes)
    _num_lanes = _num_paved_lanes + _num_ditch_lanes
    _max_lane_idx = _num_lanes - 1

    # Stage definitions
    _stage_offset = 1
    _wall_columns = np.array([0, _num_lanes + _stage_offset])
    _num_wall_columns = len(_wall_columns)
    _num_speedometer_columns = 1
    _speedometer_col_idx = -1
    _stage_width = _num_lanes + _num_wall_columns + _num_speedometer_columns

    reward_for_being_in_transit = -1

    # Cache
    _available_spaces_given_distance = {}

    @classmethod
    def available_spaces(cls, distance):
        if distance not in cls._available_spaces_given_distance:
            cls._available_spaces_given_distance[distance] = set(
                product(range(distance), range(cls._num_lanes)))
        return cls._available_spaces_given_distance[distance]

    def __init__(self,
                 headlight_range,
                 car,
                 obstacles=[],
                 allowed_obstacle_appearance_columns=None):
        self._headlight_range = headlight_range

        if self.speed_limit() < car.speed:
            raise ValueError(
                "Car's speed, {}, breaks the speed limit, {}.".format(
                    car.speed, self.speed_limit()))
        self._car = car
        self._obstacles = obstacles

        if allowed_obstacle_appearance_columns is None:
            allowed_obstacle_appearance_columns = [
                set(range(self._num_lanes)) for _ in range(len(obstacles))
            ]
        assert len(allowed_obstacle_appearance_columns) == len(obstacles)
        self._allowed_obstacle_appearance_columns = allowed_obstacle_appearance_columns

    @property
    def car(self):
        return self._car

    @property
    def obstacles(self):
        return self._obstacles

    def __eq__(self, other):
        return (self._headlight_range == other._headlight_range
                and self._car == other._car
                and self._obstacles == other._obstacles
                and (self._allowed_obstacle_appearance_columns ==
                     other._allowed_obstacle_appearance_columns))

    def copy(self):
        return self.__class__(
            self._headlight_range,
            self._car,
            self._obstacles,
            allowed_obstacle_appearance_columns=(
                self._allowed_obstacle_appearance_columns))

    def car_row(self):
        return self._headlight_range

    def _num_rows(self):
        return self._headlight_range + 1

    def speed_limit(self):
        '''The hard speed limit on this road.

        Taking the `UP` action when traveling at the speed limit has no effect.

        Set according to the headlight range since overdriving the
        headlights too much breaks the physical plausibility of the game
        due to the way obstacles are reused to simulate arbitrarily long
        roads with many obstacles. This is not too much of a restriction
        though because even overdriving the headlights by one unit is
        completely unsafe.
        '''
        return self._headlight_range + 1

    def obstacle_outside_car_path(self, obstacle):
        return (obstacle.col < 0 or obstacle.col >= self._num_lanes
                or obstacle.row > self._headlight_range)

    def is_valid_configuration(self, revealed):
        for obstacle_idx, (row, col) in revealed.items():
            if col not in self._allowed_obstacle_appearance_columns[obstacle_idx]:  # yapf:disable
                return False
        return True

    def every_combination_of_revealed_obstacles(self, distance):
        hidden_obstacle_indices = [
            i for i in range(len(self._obstacles))
            if self.obstacle_outside_car_path(self._obstacles[i])
        ]

        for num_newly_visible_obstacles in range(len(hidden_obstacle_indices) + 1):  # yapf:disable
            for positions, group in (
                permutation_combination_pairs(
                    self.available_spaces(distance),
                    hidden_obstacle_indices,
                    num_newly_visible_obstacles
                )
            ):  # yapf:disable
                revealed = dict(zip(group, positions))
                if self.is_valid_configuration(revealed):
                    yield revealed

    def successors(self, action):
        '''Generates successor, probability tuples.

        Yield:
            Road: A successor state.
            float: The probability of transitioning to the given successor
                   state.
        '''
        distance = self._car.progress_toward_destination(action)
        next_car = self._car.next(action, self.speed_limit())
        if self.has_crashed(next_car):
            next_car.speed = 0

        for revealed in self.every_combination_of_revealed_obstacles(distance):
            prob = 1.0
            next_obstacles = []

            for i in range(len(self._obstacles)):
                obs = self._obstacles[i]
                p = self.prob_obstacle_appears(obs, distance)
                assert 0 <= p <= 1

                obs_is_revealed = i in revealed
                if obs_is_revealed:
                    next_obstacle = obs.copy_at_position(*revealed[i])
                    prob_not_appearing_closer = ((1.0 - p)**(
                        distance - next_obstacle.row - 1))
                    prob_appearing_in_row = p * prob_not_appearing_closer
                    prob *= prob_appearing_in_row / float(len(self._allowed_obstacle_appearance_columns[i]))
                else:
                    next_obstacle = obs.next(distance)
                    prob *= (1.0 - p)**distance
                next_obstacles.append(next_obstacle)

            next_road = self.__class__(
                self._headlight_range,
                next_car,
                obstacles=next_obstacles,
                allowed_obstacle_appearance_columns=(
                    self._allowed_obstacle_appearance_columns))
            yield next_road, prob

    def is_off_road(self, car=None):
        if car is None: car = self._car
        return car.col <= 0 or car.col >= self._max_lane_idx

    def has_crashed(self, car=None):
        if car is None: car = self._car
        return car.col < 0 or car.col > self._max_lane_idx

    def to_key(self):
        obstacles = []
        for o in self._obstacles:
            if self.obstacle_is_visible(o):
                obstacles.append((str(o), o.row, o.col, o.speed))
        return (self._car.col, self._car.speed, frozenset(obstacles))

    def to_s(self):
        s = np.concatenate(
            [
                self.board(),
                np.full([self._num_rows(), 1], _byte('\n'), dtype='uint8')
            ],
            axis=1).tostring().decode('ascii')
        return s[:-1]

    def prob_obstacle_appears(self, obstacle, distance):
        an_obstacle_could_appear = distance > 0
        this_obstacle_could_appear = (an_obstacle_could_appear and
                                      self.obstacle_outside_car_path(obstacle))
        return (obstacle.prob_of_appearing
                if this_obstacle_could_appear else 0)

    def car_layer(self):
        layer = np.full([self._num_rows(), self._stage_width], False)
        layer[self.car_row(), self._car.col + self._stage_offset] = True
        return layer

    def wall_layer(self):
        layer = np.full([self._num_rows(), self._stage_width], False)
        layer[:, self._wall_columns] = True
        return layer

    def ditch_layer(self):
        layer = np.full([self._num_rows(), self._stage_width], False)
        layer[:, self._ditch_lanes + self._stage_offset] = True
        return layer

    def speedometer_layer(self):
        layer = np.full([self._num_rows(), self._stage_width], False)
        layer[self.speed_limit() - self._car.speed:,
              self._speedometer_col_idx] = True
        return layer

    def obstacle_layers(self):
        layers = {}
        for o in self._obstacles:
            c = str(o)
            if c not in layers:
                layers[c] = np.full([self._num_rows(), self._stage_width],
                                    False)
            if self.obstacle_is_visible(o):
                layers[c][o.row, o.col + self._stage_offset] = True
        return layers

    def layers(self):
        layers = self.obstacle_layers()
        layers[str(self._car)] = self.car_layer()
        layers['|'] = self.wall_layer()
        layers['d'] = self.ditch_layer()
        layers['^'] = self.speedometer_layer()

        full_layer = np.full([self._num_rows(), self._stage_width], False)
        for l in layers.values():
            np.logical_or(full_layer, l, out=full_layer)
        layers[' '] = np.logical_not(full_layer)

        return layers

    def ordered_layers(self):
        layers = (
            [
                (' ', np.full([self._num_rows(), self._stage_width], False)),
                ('|', self.wall_layer()),
                ('d', self.ditch_layer()),
                ('^', self.speedometer_layer())
            ]
            + list(self.obstacle_layers().items())
        )  # yapf:disable
        layers.append((str(self._car), self.car_layer()))

        for i in range(1, len(layers)):
            np.logical_or(layers[0][1], layers[i][1], out=layers[0][1])
        np.logical_not(layers[0][1], out=layers[0][1])

        return layers

    def board(self):
        board = np.zeros([self._num_rows(), self._stage_width], dtype='uint8')
        for c, layer in self.ordered_layers():
            partial = np.multiply(_byte(c), layer, dtype='uint8')
            is_present = partial > 0
            board[is_present] = partial[is_present]
        return board

    def observation(self):
        return Observation(self.board(), self.layers())

    def sample_transition(self, a):
        v = np.random.uniform()
        cumulative_p = 0.0
        for s, p in self.successors(a):
            cumulative_p += p
            if cumulative_p > v:
                return s
        return s

    def obstacle_is_visible(self, obs):
        return not self.obstacle_outside_car_path(obs) and obs.row >= 0
