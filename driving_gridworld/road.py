from itertools import product, permutations

from .obstacles import Bump, Pedestrian
from .car import car_row, car_row_array


def valid_meta_configurations(num_rows, num_bumps, num_pedestrians,
                              speed_limit):
    assert num_rows > 1
    num_rows_above_car = num_rows - 1
    num_columns = 4
    num_spaces = num_rows_above_car * num_columns
    for speed in range(speed_limit):
        for car_position in range(num_columns):
            for num_present_bumps in range(num_bumps + 1):
                for num_present_pedestrians in range(num_pedestrians + 1):
                    if (num_present_bumps + num_present_pedestrians <=
                            num_spaces):
                        yield (speed, car_position, num_present_bumps,
                               num_present_pedestrians)


def determinstic_state_component_generator(num_rows, num_bumps,
                                           num_pedestrians, speed_limit):
    assert num_rows > 1
    num_rows_above_car = num_rows - 1
    num_columns = 4
    available_columns = {}
    for i in range(num_rows_above_car):
        for j in range(num_columns):
            available_columns[(i, j)] = True

    def legal_positions():
        return available_columns.keys()

    for (speed, car_position, num_present_bumps,
         num_present_pedestrians) in valid_meta_configurations(
             num_rows, num_bumps, num_pedestrians, speed_limit):
        bump_positions = []
        pedestrian_positions = []
        for bump_positions in combinations(legal_positions(),
                                           num_present_bumps):
            for pos in bump_positions:
                del available_columns[pos]
            for pedestrian_positions in combinations(legal_positions(),
                                                     num_present_pedestrians):
                yield (speed, num_rows, bump_positions, pedestrian_positions,
                       car_position)
            for pos in bump_positions:
                available_columns[pos] = True


def determinstic_state_generator(num_rows, num_bumps, num_pedestrians,
                                 speed_limit):
    for (speed, num_rows, bump_positions, pedestrian_positions,
         car_position) in determinstic_state_component_generator(
             num_rows, num_bumps, num_pedestrians, speed_limit):
        yield speed, road_state(num_rows, bump_positions, pedestrian_positions,
                                car_position)


def bump_indices(num_bumps):
    assert num_bumps >= 0
    return ''.join([str(i) for i in range(num_bumps)])


def pedestrian_indices(num_pedestrians, num_bumps):
    assert num_bumps >= 0
    assert num_pedestrians >= 0
    return ''.join([str(i + num_bumps) for i in range(num_pedestrians)])


def road_state(num_rows,
               bump_positions=[],
               pedestrian_positions=[],
               car_position=2,
               show_walls=True):
    assert num_rows > 1

    if show_walls:
        board = (
            [['+', 'd', ' ', ' ', 'd', ' ']] + [[' ', 'd', ' ', ' ', 'd', ' ']
                                                for _ in range(num_rows - 2)] +
            [car_row_array(car_position, show_walls=show_walls)])
    else:
        board = ([['d', ' ', ' ', 'd']] + [['d', ' ', ' ', 'd']
                                           for _ in range(num_rows - 2)] +
                 [car_row_array(car_position, show_walls=show_walls)])
    for i, j in bump_positions:
        col = j + int(show_walls)
        if i < len(board) and 0 <= col <= len(board[i]) - int(show_walls):
            board[i][col] = 'b'
    for i, j in pedestrian_positions:
        col = j + int(show_walls)
        if i < len(board) and 0 <= col <= len(board[i]) - int(show_walls):
            board[i][col] = 'p'
    return '\n'.join([''.join(row) for row in board])


def road_art(num_rows, num_bumps, num_pedestrians):
    '''
    Legend:
        ' ': pavement.                    'd': dirt ditch.
        'b': bump.                        'p': pedestrian.
        'C': the player's car.
    '''
    assert num_rows > 0
    assert num_bumps >= 0
    assert num_pedestrians >= 0

    wall_to_wall_width = 6
    max_width = max(num_bumps, num_pedestrians, wall_to_wall_width)
    return (
        [
            bump_indices(num_bumps) + ' ' * (max_width - num_bumps),
            (pedestrian_indices(num_pedestrians, num_bumps) + ' ' *
             (max_width - num_pedestrians)), '+d  d ' + ' ' *
            (max_width - wall_to_wall_width)
        ] + [' d  d ' + ' ' * (max_width - wall_to_wall_width)] *
        (num_rows - 2) + [car_row() + ' ' * (max_width - wall_to_wall_width)])


def combinations(iterable, r, collection=tuple):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
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


class Road(object):
    def __init__(self, num_rows, car, obstacles, speed_limit):
        if speed_limit < car.speed:
            raise ValueError("Car's speed above speed limit!")
        self._num_rows = num_rows
        self._num_columns = 4
        self._car = car
        self._speed_limit = speed_limit
        self._obstacles = obstacles
        self._available_spaces = {}
        for pos in product(range(0, self._car.speed), range(4)):
            self._available_spaces[pos] = True
        for obs in self._obstacles:
            if not self.obstacle_outside_car_path(obs):
                disallowed_position = obs.position()
                if disallowed_position in self._available_spaces:
                    del self._available_spaces[disallowed_position]

    def obstacle_outside_car_path(self, obstacle):
        return (obstacle.col < 0 or obstacle.col >= self._num_columns
                or obstacle.row >= self._num_rows)

    def every_combination_of_revealed_obstacles(self):
        hidden_obstacle_indices = [
            i for i in range(len(self._obstacles))
            if self.obstacle_outside_car_path(self._obstacles[i])
        ]
        max_num_revealed_obstacles = min(
            len(hidden_obstacle_indices), len(self._available_spaces))

        for num_newly_visible_obstacles in range(
                max_num_revealed_obstacles + 1):
            position_sets = list(
                permutations(self._available_spaces.keys(),
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

        next_car = self._car.next(action, self._speed_limit)

        for positions, reveal_indices in (
                self.every_combination_of_revealed_obstacles()):
            prob = 1.0
            num_obstacles_revealed = 0
            next_obstacles = []
            reward = 0.0
            for i in range(len(self._obstacles)):
                obs = self._obstacles[i]
                p = self.prob_obstacle_appears(obs, num_obstacles_revealed)
                assert p <= 1
                if i in reveal_indices:
                    next_obstacle = obs.__class__(
                        *positions[num_obstacles_revealed])
                    num_avail_spaces_given_revealed_obs = (
                        len(self._available_spaces) - num_obstacles_revealed)
                    prob *= p / float(num_avail_spaces_given_revealed_obs)
                else:
                    next_obstacle = obs.next(self._car)
                    prob *= 1.0 - p
                next_obstacles.append(next_obstacle)
                reward += next_obstacle.reward(self._car)
            reward += self._car.reward()
            if self._car.col == 0 or self._car.col == 3:
                reward += -4 * self._car.speed
            next_road = self.__class__(self._num_rows, next_car,
                                       next_obstacles, self._speed_limit)
            yield (next_road, prob, reward)

    def to_key(self, show_walls=False):
        return (self.to_s(show_walls=show_walls), self._car.speed)

    def to_s(self, show_walls=True):
        bumps = [
            obstacle.position() for obstacle in self._obstacles
            if isinstance(obstacle, Bump)
        ]
        pedestrians = [
            obstacle.position() for obstacle in self._obstacles
            if isinstance(obstacle, Pedestrian)
        ]
        return road_state(
            self._num_rows,
            bumps,
            pedestrians,
            car_position=self._car.col,
            show_walls=show_walls)

    def prob_obstacle_appears(self, obstacle, num_obstacles_revealed):
        if self.obstacle_outside_car_path(obstacle):
            space_is_available = (num_obstacles_revealed < len(
                self._available_spaces))
            return obstacle.prob_of_appearing() * int(space_is_available)
        else:
            return 0
