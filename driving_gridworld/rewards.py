import numpy as np
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian


def sample_reward_bias():
    return np.random.uniform(-1, 1)


def sample_reward_parameters(headlight_range):
    u_vec = [0]
    d_vec = [np.random.uniform(-1, 0)]
    H = np.zeros(shape=(headlight_range+2, headlight_range+1), dtype=float)
    C = np.zeros(shape=(headlight_range+2, headlight_range+1), dtype=float)

    C[0, 0] = np.random.uniform(-1, u_vec[0])
    H[0, 0] = np.random.uniform(-1, min(C[0, 0], d_vec[0]))

    for i in range(1, headlight_range + 2):
        ui = np.random.uniform(u_vec[i-1], 1)
        u_vec.append(ui)
        di_min = np.random.uniform(-1, d_vec[i-1])
        di = np.random.uniform(di_min, u_vec[i])
        d_vec.append(di)
        Cmin = np.random.uniform(-1, min(C[i-1, 0], u_vec[i]))
        C[i, 0] = np.random.uniform(Cmin, u_vec[i])
        Hmin = np.random.uniform(-1, min(C[i, 0], H[i-1, 0], d_vec[i]))
        H[i, 0] = np.random.uniform(Hmin, min(C[i, 0], d_vec[i]))

    for j in range(1, headlight_range + 1):
        C[0, j] = np.random.uniform(-1, C[0, j-1])
        H[0, j] = np.random.uniform(-1, min(C[0, j], H[0, j-1]))

    for i in range(1, headlight_range + 2):
        for j in range(1, headlight_range + 1):
            Cmin = np.random.uniform(-1, min(C[i-1, j], C[i, j-1]))
            C[i, j] = np.random.uniform(Cmin, C[i, j-1])
            Hmin = np.random.uniform(-1, min(C[i, j], H[i-1, j], H[i, j-1]))
            H[i, j] = np.random.uniform(Hmin, min(C[i, j], H[i, j-1]))
    return u_vec, C, d_vec, H


def r(u, C, d, H, s, a, s_prime):
    min_col = min(s.car.col, s_prime.car.col)
    max_col = max(s.car.col, s_prime.car.col)

    def obstacle_could_encounter_car(obs, obs_prime):
        return (min_col <= obs_prime.col <= max_col
                and max(obs.row, 0) <= s.car_row() <= obs_prime.row)

    collided_obstacles_speed = []
    for obs, obs_prime in zip(s.obstacles, s_prime.obstacles):
        if obstacle_could_encounter_car(obs, obs_prime):
            if isinstance(obs, Pedestrian):
                return -1
            else:
                collided_obstacles_speed.append(obs.speed)

    distance = s.car.progress_toward_destination(a)
    car_ends_up_off_the_road = 1 <= s_prime.car.col <= 2
    return (
        (
            C[distance, :] if car_ends_up_off_the_road else H[distance, :]
        ).take(collided_obstacles_speed).sum()
        if len(collided_obstacles_speed) > 0
        else u[distance] if car_ends_up_off_the_road else d[distance]
    )


class DeterministicReward(object):
    def __init__(self, u, C, d, H):
        self.u = u
        self.c = C
        self.d = d
        self.h = H
        self.bias = sample_reward_bias()

    def __call__(self, s, a, s_p):
        reward = r(self.u, self.c, self.d, self.h, s, a, s_p)
        return reward + self.bias


class StochasticReward(object):
    def __init__(self):
        self.bias = sample_reward_bias()

    def __call__(self, s, a, s_p):
        reward = r(*sample_reward_parameters(s._headlight_range), s, a, s_p)
        return reward + self.bias
