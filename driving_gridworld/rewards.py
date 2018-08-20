import numpy as np
from itertools import product
from driving_gridworld.obstacles import Pedestrian


def sample_reward_bias():
    return np.random.uniform(-1, 1)


def sample_reward_parameters(speed_limit):
    u_vec = np.zeros(speed_limit + 1)
    d_vec = np.random.uniform(-1, 0, size=speed_limit + 1)
    C = np.random.uniform(-1, u_vec[0], size=(speed_limit + 1, speed_limit))
    H = np.random.uniform(
        -1, min(C[0, 0], d_vec[0]), size=(speed_limit + 1, speed_limit))

    for i in range(1, speed_limit + 1):
        ui = np.random.uniform(u_vec[i - 1], 1)
        u_vec[i] = ui
        di_min = np.random.uniform(-1, d_vec[i - 1])
        di = np.random.uniform(di_min, u_vec[i])
        d_vec[i] = di
        Cmin = np.random.uniform(-1, min(C[i - 1, 0], u_vec[i]))
        C[i, 0] = np.random.uniform(Cmin, u_vec[i])
        Hmin = np.random.uniform(-1, min(C[i, 0], H[i - 1, 0], d_vec[i]))
        H[i, 0] = np.random.uniform(Hmin, min(C[i, 0], d_vec[i]))
    for j in range(1, speed_limit):
        C[0, j] = np.random.uniform(-1, C[0, j - 1])
        H[0, j] = np.random.uniform(-1, min(C[0, j], H[0, j - 1]))
    for i, j in product(range(1, speed_limit + 1), range(1, speed_limit)):
        Cmin = np.random.uniform(-1, min(C[i - 1, j], C[i, j - 1]))
        C[i, j] = np.random.uniform(Cmin, C[i, j - 1])
        Hmin = np.random.uniform(-1, min(C[i, j], H[i - 1, j], H[i, j - 1]))
        H[i, j] = np.random.uniform(Hmin, min(C[i, j], H[i, j - 1]))
    return u_vec, C, d_vec, H


def r(u, C, d, H, reward_for_critical_error, s, a, s_prime):
    if s.has_crashed() or s_prime.has_crashed():
        return reward_for_critical_error

    min_col = min(s.car.col, s_prime.car.col)
    max_col = max(s.car.col, s_prime.car.col)

    def obstacle_could_encounter_car(obs, obs_prime):
        return (min_col <= obs_prime.col <= max_col
                and max(obs.row, 0) <= s.car_row() <= obs_prime.row)

    collided_obstacles_speed = []
    for obs, obs_prime in zip(s.obstacles, s_prime.obstacles):
        if obstacle_could_encounter_car(obs, obs_prime):
            if isinstance(obs, Pedestrian):
                return reward_for_critical_error
            else:
                collided_obstacles_speed.append(obs.speed)

    distance = s.car.progress_toward_destination(a)
    car_ends_up_on_pavement = 1 <= s_prime.car.col <= 2

    if car_ends_up_on_pavement:
        without_collision = u
        with_collision = C - np.expand_dims(u, axis=1)
    else:
        without_collision = d
        with_collision = H - np.expand_dims(d, axis=1)

    return (without_collision[distance] +
            with_collision[distance, :].take(collided_obstacles_speed).sum())


class DeterministicReward(object):
    @classmethod
    def unshifted(cls, *args, **kwargs):
        return cls(*args, **kwargs, b=0.0)

    def __init__(self, u, C, d, H, b=None):
        if b is None:
            b = sample_reward_bias()
        self.b = b
        self.u = u + b
        self.c = C + b
        self.d = d + b
        self.h = H + b
        self.reward_for_critical_error = b - 1

    def __call__(self, s, a, s_p):
        reward = r(self.u, self.c, self.d, self.h,
                   self.reward_for_critical_error, s, a, s_p)
        return reward


class StochasticReward(object):
    @classmethod
    def unshifted(cls, *args, **kwargs):
        return cls(*args, **kwargs, b=0.0)

    def __init__(self, b=None):
        if b is None:
            b = sample_reward_bias()
        self.b = b

    @property
    def reward_for_critical_error(self):
        return self.b - 1

    def __call__(self, s, a, s_p):
        params = [
            v + self.b for v in sample_reward_parameters(s.speed_limit())
        ]
        reward = r(*params, self.reward_for_critical_error, s, a, s_p)
        return reward
