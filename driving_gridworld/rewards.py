import numpy as np
import tensorflow as tf
from itertools import product
from driving_gridworld.obstacles import Pedestrian, Bump
import sys

DEFAULT_EPSILON = 1e-10


class PedestrianHit(RuntimeError):
    pass


class _SituationalReward(object):
    def __init__(self,
                 wc_non_critical_error_reward=-1.0,
                 stopping_reward=0,
                 epsilon=DEFAULT_EPSILON,
                 critical_error_reward=-10.0,
                 bc_unobstructed_progress_reward=1.0,
                 num_samples=1):
        self.num_samples = num_samples
        self.wc_non_critical_error_reward = wc_non_critical_error_reward
        self.stopping_reward = stopping_reward
        self.epsilon = epsilon
        self.critical_error_reward = critical_error_reward
        self.bc_unobstructed_progress_reward = bc_unobstructed_progress_reward

    def unobstructed_reward(self, progress_made):
        if progress_made < 1:
            return self.stopping_reward
        else:
            return (
                self.unobstructed_reward_between(
                    self.unobstructed_reward(progress_made - 1) + self.epsilon,
                    progress_made * self.bc_unobstructed_progress_reward
                )
            )  # yapf:disable

    def offroad_reward(self, progress_made):
        min_r = (
            self.wc_non_critical_error_reward + progress_made * self.epsilon)
        if progress_made < 0:
            return min_r
        else:
            return self.offroad_reward_between(
                min_r,
                (self.unobstructed_reward(progress_made) - self.epsilon),
                self.offroad_reward(progress_made - 1))

    def pavement_collision_reward(self, progress_made,
                                  collision_obstacle_speed):
        if progress_made < 0:
            return np.inf if self.num_samples < 2 else np.full(
                [self.num_samples], np.inf).astype('float32')
        elif collision_obstacle_speed < 0:
            return self.unobstructed_reward(progress_made)
        else:
            min_bonus = ((
                progress_made - collision_obstacle_speed) * self.epsilon)

            pcr_w_slower_obstacle = self.pavement_collision_reward(
                progress_made, collision_obstacle_speed - 1)

            bc_offroad_collision_reward = pcr_w_slower_obstacle - self.epsilon
            min_r = self._min(self.wc_non_critical_error_reward + min_bonus,
                              bc_offroad_collision_reward)

            return self.pavement_collision_reward_between(
                min_r,
                bc_offroad_collision_reward,
                pcr_w_slower_obstacle,
                self.pavement_collision_reward(progress_made - 1,
                                               collision_obstacle_speed),
            )

    def offroad_collision_reward(self,
                                 progress_made,
                                 collision_obstacle_speed,
                                 collision_obstacle_on_pavement=True):
        if not collision_obstacle_on_pavement:
            print(
                'WARNING: Case where collision obstacle is off the pavement is not currently handled by `SituationalReward`.',
                file=sys.stderr)
        if progress_made < 0:
            return np.inf if self.num_samples < 2 else np.full(
                [self.num_samples], np.inf).astype('float32')
        elif collision_obstacle_speed < 0:
            return self.offroad_reward(progress_made)
        else:
            min_bonus = ((
                progress_made - collision_obstacle_speed - 1) * self.epsilon)

            ocr_w_slower_obstacle = self.offroad_collision_reward(
                progress_made, collision_obstacle_speed - 1,
                collision_obstacle_on_pavement)
            max_r = (
                self._min(
                    self.pavement_collision_reward(progress_made,
                                                   collision_obstacle_speed),
                    ocr_w_slower_obstacle
                ) - self.epsilon
            )  # yapf:disable

            min_r = self._min(self.wc_non_critical_error_reward + min_bonus,
                              max_r)
            return self.offroad_collision_reward_between(
                min_r,
                max_r,
                ocr_w_slower_obstacle,
                self.offroad_collision_reward(progress_made - 1,
                                              collision_obstacle_speed,
                                              collision_obstacle_on_pavement),
            )

    def reward(self,
               progress_made,
               car_always_on_pavement,
               collision_obstacle_speed=None,
               collision_obstacle_on_pavement=True):
        if collision_obstacle_speed is None:
            if car_always_on_pavement:
                return self.unobstructed_reward(progress_made)
            else:
                return self.offroad_reward(progress_made)
        else:
            if car_always_on_pavement:
                return self.pavement_collision_reward(progress_made,
                                                      collision_obstacle_speed)
            else:
                return self.offroad_collision_reward(
                    progress_made, collision_obstacle_speed,
                    collision_obstacle_on_pavement)

    def __call__(self, s, a, s_prime):
        if s.has_crashed() or s_prime.has_crashed():
            return self.critical_error_reward

        distance = s.car.progress_toward_destination(a)
        car_always_on_pavement = (not (s.is_in_a_ditch()
                                       or s_prime.is_in_a_ditch()))
        reward_without_collision = self.reward(distance,
                                               car_always_on_pavement)

        def check_for_pedestrian_collision(obs):
            if isinstance(obs, Pedestrian): raise PedestrianHit()

        def check_for_bump_collision(obs):
            return ((self.reward(distance, car_always_on_pavement, obs.speed,
                                 not s.is_in_a_ditch(obs)) -
                     reward_without_collision)
                    if isinstance(obs, Bump) else None)

        try:
            rewards_collided_obstacles = s.count_obstacle_collisions(
                s_prime, check_for_pedestrian_collision,
                check_for_bump_collision)
        except PedestrianHit:
            return self.critical_error_reward

        return sum(rewards_collided_obstacles) + reward_without_collision

    def unobstructed_reward_between(self, min_reward, max_reward):
        return self._random_uniform(min_reward, max_reward)

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

    def _random_uniform(self, minval=0, maxval=1):
        return np.squeeze(
            np.random.uniform(
                low=minval, high=maxval, size=[self.num_samples]))

    def _min(self, *args):
        return np.min(args, axis=0) if self.num_samples > 1 else min(args)


class SituationalReward(_SituationalReward):
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
            self._c[progress_made].append(c)
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

    def offroad_collision_reward(self,
                                 progress_made,
                                 collision_obstacle_speed,
                                 collision_obstacle_on_pavement=True):
        if progress_made < 0 or collision_obstacle_speed < 0:
            return super().offroad_collision_reward(
                progress_made, collision_obstacle_speed,
                collision_obstacle_on_pavement)
        elif not self.offroad_collision_is_saved(progress_made,
                                                 collision_obstacle_speed):
            h = super().offroad_collision_reward(
                progress_made, collision_obstacle_speed,
                collision_obstacle_on_pavement)
            if len(self._h) <= progress_made:
                self._h.append([])
            self._h[progress_made].append(h)
        return self._h[progress_made][collision_obstacle_speed]

    def np(self, speed_limit):
        for progress_made, car_always_on_pavement, collision_obstacle_speed in product(
                range(speed_limit + 1), [True, False], range(speed_limit)):
            self.reward(progress_made, car_always_on_pavement,
                        collision_obstacle_speed)
        return (np.array(self._u), np.array(self._c), np.array(self._d),
                np.array(self._h))

    def tf(self, speed_limit):
        for progress_made, car_always_on_pavement, collision_obstacle_speed in product(
                range(speed_limit + 1), [True, False], range(speed_limit)):
            self.reward(progress_made, car_always_on_pavement,
                        collision_obstacle_speed)
        return (tf.stack(self._u), tf.stack(self._c), tf.stack(self._d),
                tf.stack(self._h))


class TfSituationalReward(SituationalReward):
    def _random_uniform(self, minval=0, maxval=1):
        return tf.squeeze(
            tf.random_uniform(
                [self.num_samples], minval=minval, maxval=maxval))

    def _min(self, *args):
        return tf.reduce_min(args, axis=0)


class WcSituationalReward(SituationalReward):
    def unobstructed_reward_between(self, min_reward, _):
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


class BcSituationalReward(SituationalReward):
    def unobstructed_reward_between(self, min_reward, max_reward):
        return max(min_reward, max_reward)

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
    return SituationalReward(
        MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS,
        REWARD_UNOBSTRUCTED_NO_PROGRESS,
        bc_unobstructed_progress_reward=MAX_REWARD_UNOBSTRUCTED_PROGRESS,
        epsilon=epsilon).np(speed_limit)


def worst_case_reward_parameters(speed_limit, epsilon=DEFAULT_EPSILON):
    return WcSituationalReward(
        MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS,
        REWARD_UNOBSTRUCTED_NO_PROGRESS,
        epsilon=epsilon).np(speed_limit)


def best_case_reward_parameters(speed_limit, epsilon=DEFAULT_EPSILON):
    return BcSituationalReward(
        MIN_REWARD_OBSTRUCTION_WITHOUT_PROGRESS,
        REWARD_UNOBSTRUCTED_NO_PROGRESS,
        bc_unobstructed_progress_reward=MAX_REWARD_UNOBSTRUCTED_PROGRESS,
        epsilon=epsilon).np(speed_limit)


def r(u, C, d, H, critical_error_reward, s, a, s_prime):
    if s.has_crashed() or s_prime.has_crashed():
        return critical_error_reward

    def check_for_pedestrian_collision(obs=None):
        if obs is None:
            return []
        elif isinstance(obs, Pedestrian):
            raise PedestrianHit()

    def check_for_bump_collision(obs=None):
        if obs is None:
            return []
        else:
            return [obs.speed] if isinstance(obs, Bump) else None

    try:
        collided_obstacles_speed = s.count_obstacle_collisions(
            s_prime, check_for_pedestrian_collision,
            check_for_bump_collision)[-1]
    except PedestrianHit:
        return critical_error_reward

    distance = s.car.progress_toward_destination(a)
    car_always_on_pavement = (not (s.is_in_a_ditch()
                                   or s_prime.is_in_a_ditch()))

    if car_always_on_pavement:
        without_collision = u[distance]
        with_collision = C[distance, :]
    else:
        without_collision = d[distance]
        with_collision = H[distance, :]

    total_with_collision = 0.0
    if len(collided_obstacles_speed) > 0:
        total_with_collision = with_collision.take(
            collided_obstacles_speed).sum()
    return total_with_collision - without_collision * (
        len(collided_obstacles_speed) - 1)


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

    def __init__(self, u, C, d, H, bias=None, critical_error_reward=-1.0):
        if bias is None:
            bias = sample_reward_bias()
        self.u = u + bias
        self.c = C + bias
        self.d = d + bias
        self.h = H + bias
        self.critical_error_reward = critical_error_reward + bias

    def __call__(self, s, a, s_p):
        reward = r(self.u, self.c, self.d, self.h, self.critical_error_reward,
                   s, a, s_p)
        return reward


class StochasticReward(object):
    @classmethod
    def unshifted(cls, *args, **kwargs):
        return cls(*args, **kwargs, bias=0.0)

    def __init__(self, bias=None, critical_error_reward=-1.0):
        if bias is None:
            bias = sample_reward_bias()
        self.bias = bias
        self.critical_error_reward = critical_error_reward + bias

    def __call__(self, s, a, s_p):
        params = [
            v + self.bias for v in sample_reward_parameters(s.speed_limit())
        ]
        reward = r(*params, self.critical_error_reward, s, a, s_p)
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
