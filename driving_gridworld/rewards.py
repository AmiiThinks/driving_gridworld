import numpy as np
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian

def bias_term():
    b = np.random.uniform(-1, 1)
    return b


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


def reward_for_collision_bump(distance, car_new_col, C, H, obst_speed):
    if 1 <= car_new_col <= 2:
        return C[distance, obst_speed]
    else:
        return H[distance, obst_speed]


def r(u, C, d, H, s, a, s_prime):
    reward = 0
    distance = s.car.progress_toward_destination(a)
    min_col = min(s.car.col, s_prime.car.col)
    max_col = max(s.car.col, s_prime.car.col)
    num_obstacles_collided = 0
    collided_obstacles_speed = list()
    for i in range(len(s._obstacles)): #suppose we only have 1 obstacle.
        obs = s._obstacles[i]
        next_obstacle = s_prime._obstacles[i]
        if (
            min_col <= next_obstacle.col <= max_col
            and max(obs.row, 0) <= s._car_row() <= next_obstacle.row
        ):  # yapf: disable
            if isinstance(next_obstacle, Pedestrian):
                return -1
            else:
                collided_obstacles_speed.append(next_obstacle.speed)
                num_obstacles_collided += 1

    assert num_obstacles_collided == len(collided_obstacles_speed)
    if num_obstacles_collided > 0:
        for obst_speed in collided_obstacles_speed:
            reward += reward_for_collision_bump(distance, s_prime.car.col, C, H, obst_speed)
    else:
        if 1 <= s_prime.car.col <= 2:
            reward += u[distance]
        else:
            reward += d[distance]
    return reward


class Deterministic_Reward(object):
    def __init__(self, u, C, d, H):
        self.u = u
        self.c = C
        self.d = d
        self.h = H
        self.bias = bias_term()

    def __call__(self, s, a, s_p):
        reward = r(self.u, self.c, self.d, self.h, s, a, s_p)
        return reward + self.bias

class Stochastic_Reward(object):
    def __init__(self):
        self.bias = bias_term()

    def __call__(self, s, a, s_p):
        num_obst = len(s._obstacles)
        hlr = s._headlight_range
        u, C, d, H = sample_reward_parameters(hlr)
        reward = r(u, C, d, H, s, a, s_p)
        return reward + self.bias
