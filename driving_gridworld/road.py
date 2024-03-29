from itertools import product, combinations
import numpy as np
from pycolab.rendering import Observation
from collections import defaultdict, UserDict
from driving_gridworld.actions import ACTIONS, NO_OP
from driving_gridworld.obstacles import Pedestrian, Bump


def _byte(c, encoding='ascii'):
    return bytes(c, encoding)[0]


def _call_or_zero(v):
    try:
        value = v()
        return 0 if value is None else value
    except:
        return 0


class Successor(object):
    def __init__(self, state=None, prob=0.0):
        self.state = state
        self.prob = prob


class Road(object):
    '''
    A gridworld that acts like a two lane road.

    One column on each side represents ditches.
    The rows are counted in graphics ordering ((0, 0) is top-left).
    The car is always on `car_row` at the bottom of the gridworld.
    The car always moves as many spaces as its speed.
    ACCELERATE and BREAK actions change the car's speed one unit
    on the next frame.
    LEFT and RIGHT move the car one space to the left or right,
    respectively.
    CRUISE is a no-op.

    Obstacles appear on the road as if they were past the headlight range
    on the previous frame.
    Obstacles are recycled when they are unobservable (the car has driven
    past them or they are not a valid column).
    '''
    # Constants
    _stage_offset = 1
    _num_speedometer_columns = 1
    _speedometer_col_idx = -1

    # Drivable road definitions
    def _init_lanes(self, paved_lanes, ditch_lanes):
        self._paved_lanes = paved_lanes
        self._ditch_lanes = ditch_lanes
        self._num_paved_lanes = len(self._paved_lanes)
        self._num_ditch_lanes = len(self._ditch_lanes)
        self._num_lanes = self._num_paved_lanes + self._num_ditch_lanes
        self._max_lane_idx = self._num_lanes - 1

        # Stage definitions
        self._wall_columns = np.array(
            [0, self._num_lanes + self._stage_offset])
        self._num_wall_columns = len(self._wall_columns)

        self._stage_width = self._num_lanes + self._num_wall_columns + self._num_speedometer_columns

    def available_spaces(self, distance):
        if distance not in self._available_spaces_given_distance:
            self._available_spaces_given_distance[distance] = set(
                product(range(distance), range(self._num_lanes)))
        return self._available_spaces_given_distance[distance]

    def __init__(self,
                 headlight_range,
                 car,
                 obstacles=[],
                 allowed_obstacle_appearance_columns=None,
                 allow_crashing=True,
                 paved_lanes=np.array([1, 2]),
                 ditch_lanes=np.array([0, 3])):
        self._init_lanes(paved_lanes, ditch_lanes)
        self._headlight_range = headlight_range
        # Cache
        self._available_spaces_given_distance = {}

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

        self.allow_crashing = allow_crashing
        if self.has_crashed():
            if not self.allow_crashing:
                raise ValueError(
                    'Car has crashed into column {} even though crashing should not be allowed.'
                    .format(self._car.col))
            self._car.speed = 0

        self._fastest_obstacle_speed = (max([o.speed for o in self._obstacles])
                                        if len(obstacles) > 0 else 0)

        def copy(headlight_range=self._headlight_range,
                 car=self._car,
                 obstacles=self._obstacles,
                 allowed_obstacle_appearance_columns=self.
                 _allowed_obstacle_appearance_columns,
                 allow_crashing=self.allow_crashing,
                 paved_lanes=self._paved_lanes,
                 ditch_lanes=self._ditch_lanes):
            return self.__class__(headlight_range,
                                  car,
                                  obstacles,
                                  allowed_obstacle_appearance_columns=
                                  allowed_obstacle_appearance_columns,
                                  allow_crashing=allow_crashing,
                                  paved_lanes=paved_lanes,
                                  ditch_lanes=ditch_lanes)

        self.copy = copy

    @property
    def car(self):
        return self._car

    @property
    def obstacles(self):
        return self._obstacles

    def __eq__(self, other):
        return (self._headlight_range == other._headlight_range
                and self._car == other._car
                and (self._allowed_obstacle_appearance_columns
                     == other._allowed_obstacle_appearance_columns)
                and (self.allow_crashing == other.allow_crashing)
                and len(self.obstacles) == len(other.obstacles) and all([
                    other.obstacles[i] == o
                    for i, o in enumerate(self.obstacles)
                ]) and (self._paved_lanes == other._paved_lanes).all()
                and (self._ditch_lanes == other._ditch_lanes).all())

    def __str__(self):
        return self.to_s()

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

    def obstacle_is_unobservable(self, obstacle):
        return (obstacle.col < 0 or obstacle.col >= self._num_lanes
                or obstacle.row > self._headlight_range)

    def is_valid_configuration(self, revealed):
        for obstacle_idx, (row, col) in revealed.items():
            if col not in self._allowed_obstacle_appearance_columns[obstacle_idx]:  # yapf:disable
                return False
        return True

    def every_combination_of_revealed_obstacles(self, distance):
        yield {}

        hidden_obstacles = []
        for i, o in enumerate(self._obstacles):
            if o.prob_of_appearing > 0 and self.obstacle_is_unobservable(o):
                spaces = [
                    (row, col)
                    for row, col in self.available_spaces(distance + o.speed)
                    if col in self._allowed_obstacle_appearance_columns[i]
                ]
                if len(spaces) > 0:
                    hidden_obstacles.append((i, spaces))

        for num_reveal in range(1, len(hidden_obstacles) + 1):
            for allocation in combinations(range(len(hidden_obstacles)),
                                           num_reveal):
                obstacles_to_reveal = [hidden_obstacles[i] for i in allocation]
                range_of_spaces = [
                    range(len(spaces)) for _, spaces in obstacles_to_reveal
                ]

                for positioning in product(*range_of_spaces):
                    yield {
                        o: spaces[space_idx]
                        for (o, spaces), space_idx in zip(
                            obstacles_to_reveal, positioning)
                    }

    def successors(self, action):
        '''Generates successor, probability tuples.

        Yield:
            Road: A successor state.
            float: The probability of transitioning to the given successor
                   state.
        '''
        d = defaultdict(Successor)
        for s, p in self._successors(action):
            d[s.to_key()].prob += p
            d[s.to_key()].state = s
        for v in d.values():
            yield v.state, v.prob

    def _successors(self, action):
        next_car = self._car.next(action, self.speed_limit())
        if self.has_crashed(next_car):
            if self.allow_crashing:
                next_car.speed = 0
                next_road = self.copy(
                    car=next_car,
                    obstacles=[o.copy() for o in self._obstacles])
                yield (next_road, 1.0)
                return
            else:
                action = NO_OP
                next_car = self._car.next(action, self.speed_limit())
        distance = self._car.progress_toward_destination(action)

        for revealed in self.every_combination_of_revealed_obstacles(distance):
            prob = 1.0
            next_obstacles = []

            obstacle_reveal_below_counts = np.zeros([
                self.speed_limit() + self._fastest_obstacle_speed,
                self._num_lanes
            ],
                                                    dtype=int)
            obstacle_reveal_above_counts = np.zeros([
                self.speed_limit() + self._fastest_obstacle_speed,
                self._num_lanes
            ],
                                                    dtype=int)
            revealed_positions = set()
            is_valid = True

            for i, obs in enumerate(self._obstacles):
                obs_distance_rt_car = distance + obs.speed

                if i in revealed:
                    if revealed[i] in revealed_positions:
                        is_valid = False
                        break
                    else:
                        revealed_positions.add(revealed[i])
                        next_obstacle = obs.copy_at_position(*revealed[i])

                        num_skipped_rows = (
                            obs_distance_rt_car - next_obstacle.row - 1 -
                            obstacle_reveal_below_counts[revealed[i]])
                        prob_not_appearing = 1.0 - obs.prob_of_appearing
                        prob_not_appearing_closer = prob_not_appearing**num_skipped_rows

                        prob_appearing_in_row = obs.prob_of_appearing * prob_not_appearing_closer

                        num_rows_below_horizon = next_obstacle.row + 1

                        num_possible_rows = sum([
                            int(obstacle_reveal_above_counts[next_obstacle.row, col] < num_rows_below_horizon)
                            for col in self._allowed_obstacle_appearance_columns[i]
                        ])  # yapf:disable
                        prob *= prob_appearing_in_row / float(
                            num_possible_rows)

                        obstacle_reveal_below_counts[:num_rows_below_horizon, next_obstacle.col] += 1  # yapf:disable
                        obstacle_reveal_above_counts[next_obstacle.row:, next_obstacle.col] += 1  # yapf:disable
                else:
                    next_obstacle = obs.next(distance)

                    if (
                        obs.prob_of_appearing > 0
                        and (
                            obs_distance_rt_car > 0
                            and self.obstacle_is_unobservable(obs)
                            and any([
                                int(obstacle_reveal_above_counts[obs_distance_rt_car - 1, col] < obs_distance_rt_car)
                                for col in self._allowed_obstacle_appearance_columns[i]
                            ])
                        )
                    ):  # yapf:disable
                        prob *= (
                            (1.0 - obs.prob_of_appearing)**obs_distance_rt_car
                        )  # yapf:disable
                if not (prob > 0):
                    is_valid = False
                    break
                next_obstacles.append(next_obstacle)

            if is_valid:
                next_road = self.copy(car=next_car, obstacles=next_obstacles)
                yield next_road, prob

    def is_in_a_ditch(self, o=None):
        if o is None: o = self._car
        for l in self._ditch_lanes:
            if o.col == l:
                return True
        return False

    def is_off_road(self, o=None):
        if o is None: o = self._car
        return self.is_in_a_ditch(o) or self.has_crashed(o)

    def has_crashed(self, car=None):
        if car is None: car = self._car
        return car.col < 0 or car.col > self._max_lane_idx

    def to_key(self):
        if self.has_crashed():
            return (-1, 0, frozenset())
        else:
            obstacles = []
            dup_count = defaultdict(lambda: 0)
            for o in self._obstacles:
                if self.obstacle_is_visible(o):
                    k = (str(o), o.row, o.col, o.speed)
                    k_prime = k + (dup_count[k], )
                    dup_count[k] += 1
                    obstacles.append(k_prime)
            return (self._car.col, self._car.speed, frozenset(obstacles))

    def to_s(self):
        s = np.concatenate([
            self.board(),
            np.full([self._num_rows(), 1], _byte('\n'), dtype='uint8')
        ],
                           axis=1).tostring().decode('ascii')
        return s[:-1]

    def prob_obstacle_appears(self, obstacle, distance):
        this_obstacle_could_appear = ((distance + obstacle.speed) > 0 and
                                      self.obstacle_is_unobservable(obstacle))
        return int(this_obstacle_could_appear) * obstacle.prob_of_appearing

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
        layers = [
            (' ', np.full([self._num_rows(), self._stage_width], False)),
            ('|', self.wall_layer()),
            ('d', self.ditch_layer()),
            ('^', self.speedometer_layer()),
            (str(self._car), self.car_layer()),
        ] + list(self.obstacle_layers().items())

        for i in range(1, len(layers)):
            np.logical_or(layers[0][1], layers[i][1], out=layers[0][1])
        np.logical_not(layers[0][1], out=layers[0][1])

        return layers

    def board(self):
        board = np.full([self._num_rows(), self._stage_width],
                        _byte(' '),
                        dtype='uint8')
        if not self.has_crashed():
            for c, layer in self.ordered_layers():
                partial = np.multiply(_byte(c), layer, dtype='uint8')
                is_present = partial > 0
                board[is_present] = partial[is_present]
        return board

    def observation(self):
        return Observation(self.board(),
                           {} if self.has_crashed() else self.layers())

    def sample_transition(self, a):
        v = np.random.uniform()
        cumulative_p = 0.0
        for s, p in self.successors(a):
            cumulative_p += p
            if cumulative_p > v: return s
        return s

    def obstacle_is_visible(self, obs):
        return not self.obstacle_is_unobservable(obs) and obs.row >= 0

    class IndexMap(UserDict):
        def __getitem__(self, item):
            if item not in self:
                self.__setitem__(item, len(self))
            return super().__getitem__(item)

    def tabulate(self, *reward_functions, print_every=None):
        transitions = []
        visited = set()
        state_indices = self.IndexMap()
        rewards = []
        to_be_visited = [(self, self.to_key())]
        while to_be_visited:
            s, s_key = to_be_visited.pop()

            if s_key in visited: continue
            visited.add(s_key)

            s_i = state_indices[s_key]

            while len(rewards) <= s_i:
                if len(reward_functions) > 1:
                    rewards_for_state = [[0.0] * len(reward_functions)
                                         for _ in range(len(ACTIONS))]
                else:
                    rewards_for_state = [0.0] * len(ACTIONS)
                rewards.append(rewards_for_state)
            while len(transitions) <= s_i:
                transitions.append([[] for _ in range(len(ACTIONS))])

            for j, a in enumerate(ACTIONS):
                sum_r = [0.0] * len(reward_functions)
                for i, (s_prime, p) in enumerate(s.successors(a)):
                    s_prime_key = s_prime.to_key()
                    if s_prime_key not in visited:
                        to_be_visited.append((s_prime, s_prime_key))

                    s_prime_i = state_indices[s_prime_key]
                    while len(transitions[s_i][j]) <= s_prime_i:
                        transitions[s_i][j].append(0.0)
                    transitions[s_i][j][s_prime_i] = p

                    for k, r in enumerate(reward_functions):
                        sum_r[k] += r(s, a, s_prime) * p

                if len(reward_functions) > 1:
                    for k in range(len(reward_functions)):
                        rewards[s_i][j][k] = sum_r[k]
                elif len(reward_functions) > 0:
                    rewards[s_i][j] = sum_r[0]

            if print_every is not None and len(visited) % print_every == 0:
                print('{} / {} visited'.format(len(visited),
                                               len(to_be_visited)))

        for i in range(len(state_indices)):
            for j in range(len(ACTIONS)):
                while len(transitions[i][j]) < len(state_indices):
                    transitions[i][j].append(0.0)

        return transitions, rewards, state_indices

    def count_obstacle_collisions(self, s_prime, *value_for_obs):
        '''
        Returns the cumulative value of colliding with each obstacle.

        Returns one value for each value function in value_for_obs.

        To collide with an obstacle, the obstacle must start in a
        position that is not the same as the car's and cross paths
        with one of the potential paths the car could take.

        Parameters:
        - s_prime: The next road state.
        - value_for_obs: A list of value functions.
            Each value function must return a value that implements
            the '+' operator given an obstacle.
            None is replaced with 0.
        '''
        counts = [_call_or_zero(v) for v in value_for_obs]

        for obs, obs_prime in zip(self.obstacles, s_prime.obstacles):
            car_has_collided = (
                # The car has driven up to or past the obstacle's row
                obs_prime.row >= self.car_row()
                # Obstacle did not start in the same position as the car
                and (obs.row != self.car_row() or obs.col != self.car.col)
                # Obstacle is in a column where a collision could have occurred
                and min(self.car.col, s_prime.car.col) <= obs_prime.col <= max(
                    self.car.col, s_prime.car.col))
            if car_has_collided:
                for i, v in enumerate(value_for_obs):
                    value = v(obs)
                    counts[i] += 0 if value is None else value
        return counts

    def safety_information(self):
        '''
        Returns the safety information associated with each state-action-next
        state sequence.

        Returns:
        - An |S| x |A| x |S| x 7 information tensor. The information is
        arranged as:
            - whether or not the car ended up crashing into a wall,
            - the number of `Pedestrian` collisions,
            - the number of `Bump` collisions,
            - whether or not the car ended up off the pavement,
            - the car's speed,
            - the amount of progress the car made, and
            - the number of lanes the car moved across.
        - A `dict` mapping between state keys and the state indices, in the
        order used to construct the information tensor.
        '''
        info = []

        visited = set()
        state_indices = self.IndexMap()
        to_be_visited = [(self, self.to_key())]

        while to_be_visited:
            s, s_key = to_be_visited.pop()

            if s_key in visited: continue
            visited.add(s_key)
            s_i = state_indices[s_key]

            while len(info) <= s_i:
                info.append([[] for _ in range(len(ACTIONS))])

            for j, a in enumerate(ACTIONS):
                for i, (s_prime, p) in enumerate(s.successors(a)):
                    s_prime_key = s_prime.to_key()
                    if s_prime_key not in visited:
                        to_be_visited.append((s_prime, s_prime_key))

                    s_prime_i = state_indices[s_prime_key]

                    collision_counts = s.count_obstacle_collisions(
                        s_prime, lambda obs: 1
                        if isinstance(obs, Pedestrian) else 0, lambda obs: 1
                        if isinstance(obs, Bump) else 0)

                    sas_info = [int(s_prime.has_crashed())] + collision_counts
                    sas_info += [
                        int(s.is_in_a_ditch() or s_prime.is_in_a_ditch()),
                        s.car.speed,
                        s.car.progress_toward_destination(a),
                        abs(s_prime.car.col - s.car.col)
                    ]

                    while len(info[s_i][j]) <= s_prime_i:
                        info[s_i][j].append([0.0] * len(sas_info))

                    info[s_i][j][s_prime_i] = sas_info

        for i in range(len(state_indices)):
            for j in range(len(ACTIONS)):
                while len(info[i][j]) < len(state_indices):
                    info[i][j].append([0.0] * 7)
        return info, state_indices


def LaneAndDitchRoad(*args, **kwargs):
    '''
    Returns a gridworld that acts like a one lane road.

    Pavement on the left in column zero and ditch on the right in column one.
    '''
    return Road(*args,
                **kwargs,
                paved_lanes=np.array([0]),
                ditch_lanes=np.array([1]))
