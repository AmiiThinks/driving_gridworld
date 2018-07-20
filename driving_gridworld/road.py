from itertools import product, permutations
import numpy as np
from pycolab.rendering import Observation


def combinations(iterable, r, collection=tuple):
    '''`r`-size `collection`s of elements in `iterable`.'''
    iterable = tuple(iterable)
    n = len(iterable)
    if r > n:
        return
    indices = list(range(r))
    yield collection(iterable[i] for i in indices)
    while True:
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield collection(iterable[i] for i in indices)


def _byte(c, encoding='ascii'):
    return bytes(c, encoding)[0]


def permutation_combination_pairs(a, b, n):
    return product(
        list(permutations(a, n)), list(combinations(b, n, collection=set)))


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

    def __init__(self, headlight_range, car, obstacles=[], stddev=0.0):
        self._headlight_range = headlight_range

        if self.speed_limit() < car.speed:
            raise ValueError(
                "Car's speed, {}, breaks the speed limit, {}.".format(
                    car.speed, self.speed_limit()))
        self._car = car
        self._obstacles = obstacles
        self._available_spaces = set()
        self._stddev = stddev

    def __eq__(self, other):
        return (self._headlight_range == other._headlight_range
                and self._num_lanes == other._num_lanes
                and self._car == other._car
                and self._obstacles == other._obstacles
                and self._available_spaces == other._available_spaces
                and self._stddev == other._stddev)

    def copy(self):
        return self.__class__(
            self._headlight_range,
            self._car,
            self._obstacles,
            stddev=self._stddev)

    def _car_row(self):
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

    def every_combination_of_revealed_obstacles(self, distance):
        self._available_spaces = set()
        for pos in product(range(distance), range(self._num_lanes)):
            self._available_spaces.add(pos)

        hidden_obstacle_indices = [
            i for i in range(len(self._obstacles))
            if self.obstacle_outside_car_path(self._obstacles[i])
        ]
        max_num_revealed_obstacles = min(
            len(hidden_obstacle_indices), len(self._available_spaces))

        for num_newly_visible_obstacles in range(
                max_num_revealed_obstacles + 1):
            for positions, reveal_indices in permutation_combination_pairs(
                    self._available_spaces, hidden_obstacle_indices,
                    num_newly_visible_obstacles):
                assert len(positions) == len(reveal_indices)
                yield positions, reveal_indices

    def successors(self, action):
        '''Generates successor, probability, reward tuples.

        Yield:
            Road: A successor state.
            float: The probability of transitioning to the given successor
                   state.
            float: The reward given the current state, action, and successor
                   state. The reward function is deterministic.
        '''
        next_car = self._car.next(action, self.speed_limit())
        if self.has_crashed() or self.has_crashed(next_car):
            distance = 0
            next_car.speed = 0
        else:
            distance = self._car.progress_toward_destination(action)

        revealed_obstacles = (
            self.every_combination_of_revealed_obstacles(distance)
            if distance > 0 else [(None, set())])

        for positions, reveal_indices in revealed_obstacles:
            prob = 1.0
            num_obstacles_revealed = 0
            next_obstacles = []
            reward = self.reward_for_being_in_transit

            reward = -1.0
            for i in range(len(self._obstacles)):
                obs = self._obstacles[i]
                p = self.prob_obstacle_appears(obs, num_obstacles_revealed,
                                               distance)
                assert 0 <= p <= 1

                obs_is_revealed = i in reveal_indices
                if obs_is_revealed:
                    next_obstacle = obs.copy_at_position(
                        *positions[num_obstacles_revealed])
                    num_avail_spaces_given_revealed_obs = (
                        len(self._available_spaces) - num_obstacles_revealed)
                    prob *= p / float(num_avail_spaces_given_revealed_obs)
                    num_obstacles_revealed += 1
                else:
                    next_obstacle = obs.next(distance)
                    prob *= 1.0 - p
                next_obstacles.append(next_obstacle)

                if next_obstacle.col == next_car.col:
                    obstacle_was_in_front_of_car = (obs.row < self._car_row()
                                                    or obs_is_revealed)
                    car_ran_over_obstacle = (
                        obstacle_was_in_front_of_car
                        and next_obstacle.row >= self._car_row())
                    car_changed_lanes = self._car.col != next_car.col
                    car_changed_lanes_into_obstacle = (
                        car_changed_lanes and self._car_row() == obs.row)
                    collision_occurred = (car_changed_lanes_into_obstacle
                                          or car_ran_over_obstacle)

                    if collision_occurred:
                        reward += next_obstacle.reward_for_collision(
                            self._car.speed, self._stddev)
            reward += self._car.reward(action)

            if self.is_off_road():
                reward -= 2 * distance * self._car.speed
                noise = np.random.normal(0, self._stddev * self._car.speed)
                reward += noise
            next_road = self.__class__(self._headlight_range, next_car,
                                       next_obstacles)
            yield (next_road, prob, reward)

    def is_off_road(self):
        return self._car.col <= 0 or self._car.col >= self._max_lane_idx

    def has_crashed(self, car=None):
        if car is None: car = self._car
        return car.col < 0 or car.col > self._max_lane_idx

    def to_key(self):
        obstacles = []
        for o in self._obstacles:
            if self.obstacle_is_visible(o):
                obstacles.append((str(o), o.row, o.col))
        return (self._car.col, self._car.speed, frozenset(obstacles))

    def to_s(self):
        s = np.concatenate(
            [
                self.board(),
                np.full([self._num_rows(), 1], _byte('\n'), dtype='uint8')
            ],
            axis=1).tostring().decode('ascii')
        return s[:-1]

    def prob_obstacle_appears(self, obstacle, num_obstacles_revealed,
                              distance):
        space_is_available = (num_obstacles_revealed < len(
            self._available_spaces))
        an_obstacle_could_appear = space_is_available and distance > 0
        this_obstacle_could_appear = (an_obstacle_could_appear and
                                      self.obstacle_outside_car_path(obstacle))
        return (obstacle.prob_of_appearing
                if this_obstacle_could_appear else 0)

    def car_layer(self):
        layer = np.full([self._num_rows(), self._stage_width], False)
        layer[self._car_row(), self._car.col + self._stage_offset] = True
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
        for s, p, r in self.successors(a):
            cumulative_p += p
            if cumulative_p > v:
                return s, r
        return s, r

    def obstacle_is_visible(self, obs):
        return not self.obstacle_outside_car_path(obs) and obs.row >= 0
