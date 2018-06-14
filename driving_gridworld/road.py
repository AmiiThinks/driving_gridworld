from itertools import product, permutations
import numpy as np
from pycolab.rendering import Observation

from .car import car_row_array


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


class Road(object):
    def __init__(self, headlight_range, car, obstacles=[]):
        self._headlight_range = headlight_range

        if self.speed_limit() < car.speed:
            raise ValueError(
                "Car's speed, {}, breaks the speed limit, {}.".format(
                    car.speed, self.speed_limit()))

        self._num_lanes = 4
        self._car = car
        self._obstacles = obstacles
        self._available_spaces = set()

    def __eq__(self, other):
        return (
            self._headlight_range == other._headlight_range
            and self._num_lanes == other._num_lanes
            and self._car == other._car
            and self._obstacles == other._obstacles
            and self._available_spaces == other._available_spaces)

    def copy(self):
        return self.__class__(self._headlight_range, self._car,
                              self._obstacles)

    def _car_row(self):
        return self._headlight_range

    def _num_rows(self):
        return self._headlight_range + 1

    def speed_limit(self):
        '''The hard speed limit on this road.

        Taking the `UP` action when traveling at the speed limit has no effect.

        Set according to the headlight range since overdriving the
        headlights too much breaks the physical plausibility of the game
        due to the way we reusing obstacles to simulate arbitrarily long
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
        for obs in self._obstacles:
            if not self.obstacle_outside_car_path(obs):
                self._available_spaces.discard(obs.position())

        hidden_obstacle_indices = [
            i for i in range(len(self._obstacles))
            if self.obstacle_outside_car_path(self._obstacles[i])
        ]
        max_num_revealed_obstacles = min(
            len(hidden_obstacle_indices), len(self._available_spaces))

        for num_newly_visible_obstacles in range(
                max_num_revealed_obstacles + 1):
            position_sets = list(
                permutations(self._available_spaces,
                             num_newly_visible_obstacles))
            sets_of_obstacle_indices_to_reveal = list(
                combinations(
                    hidden_obstacle_indices,
                    num_newly_visible_obstacles,
                    collection=set))

            for positions, reveal_indices in product(
                    position_sets, sets_of_obstacle_indices_to_reveal):
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

        distance = self._car.progress_toward_destination(action)
        next_car = self._car.next(action, self.speed_limit())
        revealed_obstacles = (
            self.every_combination_of_revealed_obstacles(distance)
            if distance > 0 else [(None, set())])

        for positions, reveal_indices in revealed_obstacles:
            prob = 1.0
            num_obstacles_revealed = 0
            next_obstacles = []
            reward = -1.0
            for i in range(len(self._obstacles)):
                obs = self._obstacles[i]
                p = self.prob_obstacle_appears(obs, num_obstacles_revealed,
                                               distance)
                assert p <= 1

                obs_is_revealed = i in reveal_indices
                if obs_is_revealed:
                    next_obstacle = obs.__class__(
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
                            self._car.speed)
            reward += self._car.reward(action)

            car_is_off_road = self._car.col == 0 or self._car.col == 3
            if car_is_off_road:
                reward -= 2 * distance * self._car.speed

            next_road = self.__class__(self._headlight_range, next_car,
                                       next_obstacles)
            yield (next_road, prob, reward)

    def to_key(self):
        obstacles = []
        for o in self._obstacles:
            if self.obstacle_is_visible(o):
                obstacles.append((str(o), o.row, o.col))
        return (self._car.col, self._car.speed, frozenset(obstacles))

    def to_s(self, show_walls=True):
        template = (['|', 'd', ' ', ' ', 'd', '|']
                    if show_walls else ['d', ' ', ' ', 'd'])
        board = ([template for _ in range(self._headlight_range)] +
                 [car_row_array(self._car.col, show_walls=show_walls)])
        for obs in self._obstacles:
            if self.obstacle_is_visible(obs):
                board[obs.row, obs.col + int(show_walls)] = str(obs)
        return '\n'.join([''.join(row) for row in board])

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
        layer = np.full([self._num_rows(), self._num_lanes + 2], False)
        layer[self._car_row(), self._car.col + 1] = True
        return layer

    def wall_layer(self):
        layer = np.full([self._num_rows(), self._num_lanes + 2], False)
        layer[:, 0] = True
        layer[:, -1] = True
        return layer

    def ditch_layer(self):
        layer = np.full([self._num_rows(), self._num_lanes + 2], False)
        layer[:, 1] = True
        layer[:, -2] = True
        return layer

    def obstacle_layers(self):
        layers = {}
        for o in self._obstacles:
            c = str(o)
            if c not in layers:
                layers[c] = np.full([self._num_rows(), self._num_lanes + 2],
                                    False)
            if self.obstacle_is_visible(o):
                layers[c][o.row, o.col + 1] = True
        return layers

    def layers(self):
        layers = self.obstacle_layers()
        layers[str(self._car)] = self.car_layer()
        layers['|'] = self.wall_layer()
        layers['d'] = self.ditch_layer()

        full_layer = np.full([self._num_rows(), self._num_lanes + 2], False)
        for l in layers.values():
            np.logical_or(full_layer, l, out=full_layer)
        layers[' '] = np.logical_not(full_layer)

        return layers

    def ordered_layers(self):
        obstacle_layers = self.obstacle_layers()

        layers = [(' ', np.full([self._num_rows(), self._num_lanes + 2],
                                False)), ('|', self.wall_layer()),
                  ('d', self.ditch_layer())] + list(obstacle_layers.items())
        layers.append((str(self._car), self.car_layer()))

        for i in range(1, len(layers)):
            np.logical_or(layers[0][1], layers[i][1], out=layers[0][1])
        np.logical_not(layers[0][1], out=layers[0][1])

        return layers

    def board(self):
        board = np.zeros(
            [self._num_rows(), self._num_lanes + 2], dtype='uint8')
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
