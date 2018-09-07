import numpy as np
from itertools import product
from driving_gridworld.obstacles import Pedestrian


def sample_reward_bias():
    return np.random.uniform(-1, 1)


def specify_bounds():
    upper_bound = 1.0
    lower_bound_for_no_progress = -1.0
    return lower_bound_for_no_progress, upper_bound


def sample_reward_parameters(speed_limit, epsilon=10**-6):
    lb, ub = specify_bounds()
    mp = (lb + ub) / 2
    u_vec = np.zeros(speed_limit + 1)
    d_vec = np.random.uniform(lb, mp, size=speed_limit + 1)
    C = np.random.uniform(lb, u_vec[0], size=(speed_limit + 1, speed_limit))
    H = np.random.uniform(
        lb, min(C[0, 0], d_vec[0]), size=(speed_limit + 1, speed_limit))

    for i in range(1, speed_limit + 1):
        ui = np.random.uniform(u_vec[i - 1], ub)
        u_vec[i] = ui + epsilon
        di_min = np.random.uniform(lb, d_vec[i - 1])
        di = np.random.uniform(di_min, u_vec[i])
        d_vec[i] = di + epsilon
        Cmin = np.random.uniform(lb, min(C[i - 1, 0], u_vec[i]))
        C[i, 0] = np.random.uniform(Cmin, u_vec[i]) + epsilon
        Hmin = np.random.uniform(lb, min(C[i, 0], H[i - 1, 0], d_vec[i]))
        H[i, 0] = np.random.uniform(Hmin, min(C[i, 0], d_vec[i])) + epsilon
    for j in range(1, speed_limit):
        C[0, j] = np.random.uniform(lb, C[0, j - 1]) - epsilon
        H[0, j] = np.random.uniform(lb, min(C[0, j], H[0, j - 1])) - epsilon
    for i, j in product(range(1, speed_limit + 1), range(1, speed_limit)):
        Cmin = np.random.uniform(lb, min(C[i - 1, j], C[i, j - 1]))
        C[i, j] = np.random.uniform(Cmin, C[i, j - 1]) + ((i-j) * epsilon)
        Hmin = np.random.uniform(lb, min(C[i, j], H[i - 1, j], H[i, j - 1]))
        H[i, j] = np.random.uniform(Hmin, min(C[i, j], H[i, j - 1])) + ((i-j) * epsilon)
    return u_vec, C, d_vec, H


def sample_min_reward_parameters(speed_limit, epsilon=10**-6):
    lb, ub = specify_bounds()
    mp = (lb + ub) / 2
    u_vec = np.array([mp] * (speed_limit + 1))
    d_vec = np.array([lb] * (speed_limit + 1))
    C = np.array([[lb] * speed_limit] * (speed_limit + 1))
    H = np.array([[lb] * speed_limit] * (speed_limit + 1))
    for i in range(1, speed_limit + 1):
        u_vec[i] = u_vec[i - 1] + epsilon
        d_vec[i] = d_vec[i - 1] + epsilon
        C[i, 0] = C[i - 1, 0] + epsilon
        H[i, 0] = H[i - 1, 0] + epsilon
    for j in range(1, speed_limit):
        C[0, j] = C[0, j - 1] - epsilon
        H[0, j] = H[0, j - 1] - epsilon
    for i, j in product(range(1, speed_limit + 1), range(1, speed_limit)):
        C[i, j] = lb + ((i-j) * epsilon)
        H[i, j] = lb + ((i-j) * epsilon)
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
        return cls(*args, **kwargs, bias=0.0)

    @classmethod
    def sample(cls, speed_limit, **kwargs):
        return cls(*sample_reward_parameters(speed_limit), **kwargs)

    @classmethod
    def sample_unshifted(cls, speed_limit, **kwargs):
        return cls(*sample_reward_parameters(speed_limit), **kwargs, bias=0.0)

    @classmethod
    def sample_min_reward_unshifted(cls, speed_limit, **kwargs):
        return cls(*sample_min_reward_parameters(speed_limit), **kwargs,
            bias=0.0)

    def __init__(self, u, C, d, H, bias=None, reward_for_critical_error=-1):
        if bias is None:
            bias = sample_reward_bias()
        self.u = u + bias
        self.c = C + bias
        self.d = d + bias
        self.h = H + bias
        self.reward_for_critical_error = reward_for_critical_error + bias

    def __call__(self, s, a, s_p):
        reward = r(self.u, self.c, self.d, self.h,
                   self.reward_for_critical_error, s, a, s_p)
        return reward


class StochasticReward(object):
    @classmethod
    def unshifted(cls, *args, **kwargs):
        return cls(*args, **kwargs, bias=0.0)

    def __init__(self, bias=None, reward_for_critical_error=-1):
        if bias is None:
            bias = sample_reward_bias()
        self.bias = bias
        self.reward_for_critical_error = reward_for_critical_error + bias

    def __call__(self, s, a, s_p):
        params = [
            v + self.bias for v in sample_reward_parameters(s.speed_limit())
        ]
        reward = r(*params, self.reward_for_critical_error, s, a, s_p)
        return reward


def group_reward_parameters_for_multiple_seeds(speed_limit):
    U = np.zeros(speed_limit + 1)
    D = np.zeros(speed_limit + 1)
    H = np.zeros(speed_limit + 1)
    C = np.zeros(hspeed_limit + 1)

    initial_seed = 0
    end = 100
    step = 1
    num_samples = int((end - initial_seed)/step)

    for seed in range(initial_seed, end, step):
        np.random.seed(seed)
        u, c, d, h = sample_reward_parameters(speed_limit)
          #reward_function = StochasticReward.unshifted(reward_for_critical_error=-10.0)
        U = np.c_[U, u]
        D = np.c_[D, d]
        H = np.c_[H, h]
        C = np.c_[C, c]

    U = U[:, 1:]
    C = C[:, 1:]
    D = D[:, 1:]
    H = H[:, 1:]

    return U, C, D, H, num_samples


def average_reward_parameters(speed_limit):
    U, C, D, H, num_samples = group_reward_parameters_for_multiple_seeds(speed_limit)
    u_exp = U.mean(axis=1)
    d_exp = D.mean(axis=1)

    assert len(H[0]) == len(C[0])
    numcols = len(H[0])
    assert numcols == num_samples * (speed_limit)
    H_exp = np.zeros((speed_limit + 1, speed_limit))
    C_exp = np.zeros((speed_limit + 1, speed_limit))

    for i in range(0, speed_limit + 1):
        for initial_col in range(0, speed_limit):
          h = np.array(H[i, initial_col : numcols : speed_limit])
          H_exp[i, initial_col] = h.mean()
          c = np.array(C[i, initial_col : numcols : speed_limit])
          C_exp[i, initial_col] = c.mean()
    assert H_exp.shape == C_exp.shape

    return u_exp, d_exp, C_exp, H_exp
