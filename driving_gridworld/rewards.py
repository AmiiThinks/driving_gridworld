import numpy as np
import tensorflow as tf
from itertools import product
from driving_gridworld.obstacles import Pedestrian

DEFAULT_EPSILON = 1e-10


class SituationalReward(object):
    def __init__(self,
                 wc_non_critical_error_reward,
                 stopping_reward,
                 epsilon=DEFAULT_EPSILON):
        self.wc_non_critical_error_reward = wc_non_critical_error_reward
        self.stopping_reward = stopping_reward
        self.epsilon = epsilon

    def unobstructed_reward(self, progress_made):
        if progress_made < 1:
            return self.stopping_reward
        else:
            return (
                self.unobstructed_reward_at_least(
                    self.unobstructed_reward(progress_made - 1)
                ) + self.epsilon
            )  # yapf:disable

    def offroad_reward(self, progress_made):
        min_r = (
            self.wc_non_critical_error_reward + progress_made * self.epsilon)
        if progress_made < 0:
            return min_r
        else:
            return self.offroad_reward_between(
                min_r,
                self.unobstructed_reward(progress_made) - self.epsilon,
                self.offroad_reward(progress_made - 1))

    def pavement_collision_reward(self, progress_made,
                                  collision_obstacle_speed):
        if progress_made < 0:
            return np.inf
        elif collision_obstacle_speed < 0:
            return self.unobstructed_reward(progress_made)
        else:
            min_bonus = ((
                progress_made - collision_obstacle_speed) * self.epsilon)
            bc_offroad_collision_reward = (self.pavement_collision_reward(
                progress_made, collision_obstacle_speed - 1) - self.epsilon)
            min_r = self._min(self.wc_non_critical_error_reward + min_bonus,
                              bc_offroad_collision_reward)
            return self.pavement_collision_reward_between(
                min_r,
                bc_offroad_collision_reward,
                self.pavement_collision_reward(progress_made,
                                               collision_obstacle_speed - 1),
                self.pavement_collision_reward(progress_made - 1,
                                               collision_obstacle_speed),
            )

    def offroad_collision_reward(self, progress_made,
                                 collision_obstacle_speed):
        if progress_made < 0:
            return np.inf
        elif collision_obstacle_speed < 0:
            return self.offroad_reward(progress_made)
        else:
            min_bonus = ((
                progress_made - collision_obstacle_speed - 1) * self.epsilon)
            max_r = (
                self._min(
                    self.pavement_collision_reward(progress_made,
                                                   collision_obstacle_speed),
                    self.offroad_collision_reward(progress_made,
                                                  collision_obstacle_speed - 1)
                ) - self.epsilon
            )  # yapf:disable

            min_r = self._min(self.wc_non_critical_error_reward + min_bonus,
                              max_r)
            return self.offroad_collision_reward_between(
                min_r,
                max_r,
                self.offroad_collision_reward(progress_made,
                                              collision_obstacle_speed - 1),
                self.offroad_collision_reward(progress_made - 1,
                                              collision_obstacle_speed),
            )

    def reward(self,
               progress_made,
               car_ends_up_on_pavement,
               collision_obstacle_speed=None):
        if collision_obstacle_speed is None:
            if car_ends_up_on_pavement:
                return self.unobstructed_reward(progress_made)
            else:
                return self.offroad_reward(progress_made)
        else:
            if car_ends_up_on_pavement:
                return self.pavement_collision_reward(progress_made,
                                                      collision_obstacle_speed)
            else:
                return self.offroad_collision_reward(progress_made,
                                                     collision_obstacle_speed)

    def _min(self, *args):
        return min(args)


class CachedSituationalReward(SituationalReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._u = []
        self._d = []
        self._c = [[]]
        self._h = [[]]

    def unobstructed_reward(self, progress_made):
        if len(self._u) <= progress_made:
            self._u.append(super().unobstructed_reward(progress_made))
        return self._u[progress_made]

    def pavement_collision_is_saved(self, progress_made,
                                    collision_obstacle_speed):
        return (len(self._c) > progress_made
                and len(self._c[progress_made]) > collision_obstacle_speed)

    def pavement_collision_reward(self, progress_made,
                                  collision_obstacle_speed):
        if progress_made < 0 or collision_obstacle_speed < 0:
            return super().pavement_collision_reward(progress_made,
                                                     collision_obstacle_speed)
        elif not self.pavement_collision_is_saved(progress_made,
                                                  collision_obstacle_speed):
            c = super().pavement_collision_reward(progress_made,
                                                  collision_obstacle_speed)
            if len(self._c) <= progress_made:
                self._c.append([])
            if len(self._c[progress_made]) <= collision_obstacle_speed:
                self._c[progress_made].append(c)
            self.pavement_collision_reward(
                len(self._c) - 1,
                len(self._c[0]) - 1)
        return self._c[progress_made][collision_obstacle_speed]

    def offroad_reward(self, progress_made):
        if progress_made < 0:
            return super().offroad_reward(progress_made)
        elif len(self._d) <= progress_made:
            self._d.append(super().offroad_reward(progress_made))
        return self._d[progress_made]

    def offroad_collision_is_saved(self, progress_made,
                                   collision_obstacle_speed):
        return (len(self._h) > progress_made
                and len(self._h[progress_made]) > collision_obstacle_speed)

    def offroad_collision_reward(self, progress_made,
                                 collision_obstacle_speed):
        if progress_made < 0 or collision_obstacle_speed < 0:
            return super().offroad_collision_reward(progress_made,
                                                    collision_obstacle_speed)
        elif not self.offroad_collision_is_saved(progress_made,
                                                 collision_obstacle_speed):
            h = super().offroad_collision_reward(progress_made,
                                                 collision_obstacle_speed)
            if len(self._h) <= progress_made:
                self._h.append([])
            if len(self._h[progress_made]) <= collision_obstacle_speed:
                self._h[progress_made].append(h)
            self.offroad_collision_reward(
                len(self._h) - 1,
                len(self._h[0]) - 1)
        return self._h[progress_made][collision_obstacle_speed]

    def np(self, speed_limit):
        for progress_made, car_ends_up_on_pavement, collision_obstacle_speed in product(
                range(speed_limit + 1), [True, False], range(speed_limit)):
            self.reward(progress_made, car_ends_up_on_pavement,
                        collision_obstacle_speed)
        return (np.array(self._u), np.array(self._c), np.array(self._d),
                np.array(self._h))

    def tf(self, speed_limit):
        for progress_made, car_ends_up_on_pavement, collision_obstacle_speed in product(
                range(speed_limit + 1), [True, False], range(speed_limit)):
            self.reward(progress_made, car_ends_up_on_pavement,
                        collision_obstacle_speed)
        return (tf.stack(self._u), tf.stack(self._c), tf.stack(self._d),
                tf.stack(self._h))


class UniformSituationalReward(CachedSituationalReward):
    def __init__(self, *args, max_unobstructed_progress_reward=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_unobstructed_progress_reward = max_unobstructed_progress_reward

    def unobstructed_reward_at_least(self, min_reward):
        return self._random_uniform(min_reward,
                                    self.max_unobstructed_progress_reward)

    def offroad_reward_between(self, min_r, max_r, less_progress_reward):
        worst_wc_r = self._random_uniform(min_r,
                                          less_progress_reward + self.epsilon)
        return self._random_uniform(worst_wc_r, max_r)

    def pavement_collision_reward_between(self, min_r, max_r,
                                          less_obstacle_speed_reward,
                                          less_progress_reward):
        worst_wc_r = self._random_uniform(
            min_r,
            self._min(less_obstacle_speed_reward - self.epsilon,
                      less_progress_reward + self.epsilon, max_r))
        return self._random_uniform(worst_wc_r, max_r)

    def offroad_collision_reward_between(self, min_r, max_r,
                                         less_obstacle_speed_reward,
                                         less_progress_reward):
        return self.pavement_collision_reward_between(
            min_r, max_r, less_obstacle_speed_reward, less_progress_reward)

    def _random_uniform(self, minval=0, maxval=1, shape=[]):
        return np.random.uniform(low=minval, high=maxval, size=shape)


class TfUniformSituationalReward(UniformSituationalReward):
    def _random_uniform(self, minval=0, maxval=1, shape=[]):
        return tf.random_uniform(shape, minval=minval, maxval=maxval)

    def _min(self, *args):
        return tf.reduce_min(args)


class WcSituationalReward(CachedSituationalReward):
    def unobstructed_reward_at_least(self, min_reward):
        return min_reward

    def offroad_reward_between(self, min_r, max_r, less_progress_reward):
        return min_r

    def pavement_collision_reward_between(self, min_r, max_r,
                                          less_obstacle_speed_reward,
                                          less_progress_reward):
        return min_r

    def offroad_collision_reward_between(self, min_r, max_r,
                                         less_obstacle_speed_reward,
                                         less_progress_reward):
        return min_r


class BcUniformSituationalReward(UniformSituationalReward):
    def unobstructed_reward_at_least(self, min_reward):
        return max(min_reward, self.max_unobstructed_progress_reward)

    def offroad_reward_between(self, min_r, max_r, less_progress_reward):
        return max_r

    def pavement_collision_reward_between(self, min_r, max_r,
                                          less_obstacle_speed_reward,
                                          less_progress_reward):
        return max_r

    def offroad_collision_reward_between(self, min_r, max_r,
                                         less_obstacle_speed_reward,
                                         less_progress_reward):
        return max_r


def sample_reward_bias():
    return np.random.uniform(-1, 1)


MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS = -1.0
MAX_REWARD_UNOBSTRUCTED_PROGRESS = 1.0
REWARD_UNOBSTRUCTED_NO_PROGRESS = (
    (
        MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS
        + MAX_REWARD_UNOBSTRUCTED_PROGRESS
    ) / 2
)  # yapf:disable


def sample_reward_parameters(speed_limit, epsilon=DEFAULT_EPSILON):
    return UniformSituationalReward(
        MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS,
        REWARD_UNOBSTRUCTED_NO_PROGRESS,
        max_unobstructed_progress_reward=MAX_REWARD_UNOBSTRUCTED_PROGRESS,
        epsilon=epsilon).np(speed_limit)


def worst_case_reward_parameters(speed_limit, epsilon=DEFAULT_EPSILON):
    return WcSituationalReward(
        MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS,
        REWARD_UNOBSTRUCTED_NO_PROGRESS,
        epsilon=epsilon).np(speed_limit)


def best_case_reward_parameters(speed_limit, epsilon=DEFAULT_EPSILON):
    return BcUniformSituationalReward(
        MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS,
        REWARD_UNOBSTRUCTED_NO_PROGRESS,
        max_unobstructed_progress_reward=MAX_REWARD_UNOBSTRUCTED_PROGRESS,
        epsilon=epsilon).np(speed_limit)


def r(u, C, d, H, reward_for_critical_error, s, a, s_prime, mode='np'):
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
        with_collision = C - (tf if mode == 'tf' else np).expand_dims(
            u, axis=1)
    else:
        without_collision = d
        with_collision = H - (tf if mode == 'tf' else np).expand_dims(
            d, axis=1)

    if mode == 'tf':
        return (without_collision[distance] + tf.reduce_sum(
            tf.gather(with_collision[distance, :], collided_obstacles_speed)))
    else:
        return (
            without_collision[distance] +
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
    def worst_case_reward_unshifted(cls, speed_limit, **kwargs):
        return cls(
            *worst_case_reward_parameters(speed_limit), **kwargs, bias=0.0)

    @classmethod
    def best_case_reward_unshifted(cls, speed_limit, **kwargs):
        return cls(
            *best_case_reward_parameters(speed_limit), **kwargs, bias=0.0)

    @classmethod
    def average_reward_unshifted(cls, speed_limit, **kwargs):
        return cls(*average_reward_parameters(speed_limit), **kwargs, bias=0.0)

    @classmethod
    def tf_sample_reward_unshifted(cls, speed_limit, **kwargs):
        return cls(
            *TfUniformSituationalReward(
                MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS,
                REWARD_UNOBSTRUCTED_NO_PROGRESS,
                max_unobstructed_progress_reward=(
                    MAX_REWARD_UNOBSTRUCTED_PROGRESS),
                epsilon=DEFAULT_EPSILON).tf(speed_limit),
            **kwargs,
            bias=0.0,
            mode='tf')

    def __init__(self,
                 u,
                 C,
                 d,
                 H,
                 bias=None,
                 reward_for_critical_error=-1,
                 mode='np'):
        if bias is None:
            bias = sample_reward_bias()
        self.u = u + bias
        self.c = C + bias
        self.d = d + bias
        self.h = H + bias
        self.reward_for_critical_error = reward_for_critical_error + bias
        self.mode = mode

    def __call__(self, s, a, s_p):
        reward = r(
            self.u,
            self.c,
            self.d,
            self.h,
            self.reward_for_critical_error,
            s,
            a,
            s_p,
            mode=self.mode)
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


def sample_multiple_reward_parameters(speed_limit, num_samples=100):
    U = []
    D = []
    H = []
    C = []

    for _ in range(num_samples):
        u, c, d, h = sample_reward_parameters(speed_limit)
        U.append(u)
        D.append(d)
        H.append(h)
        C.append(c)

    U = np.array(U).T
    C = np.array(C).transpose([1, 2, 0])
    D = np.array(D).T
    H = np.array(H).transpose([1, 2, 0])

    return U, C, D, H


def average_reward_parameters(speed_limit, num_samples=100):
    return (
        v.mean(axis=-1)
        for v in sample_multiple_reward_parameters(speed_limit, num_samples))
