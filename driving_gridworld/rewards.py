import numpy as np
import tensorflow as tf
from itertools import product
from driving_gridworld.obstacles import Pedestrian, Bump


class PedestrianHit(RuntimeError):
    pass


class _SituationalReward(object):
    def __init__(self,
                 wc_non_critical_error_reward=-1.0,
                 stopping_reward=0,
                 critical_error_reward=-10.0,
                 bc_unobstructed_progress_reward=1.0,
                 num_samples=1,
                 use_slow_collision_as_offroad_base=False):
        self.num_samples = num_samples
        self.wc_non_critical_error_reward = wc_non_critical_error_reward
        self.stopping_reward = stopping_reward
        self.critical_error_reward = critical_error_reward
        self.bc_unobstructed_progress_reward = bc_unobstructed_progress_reward
        self.use_slow_collision_as_offroad_base = use_slow_collision_as_offroad_base

    def progress_bonus(self):
        return self.progress_bonus_below(self.bc_unobstructed_progress_reward)

    def offroad_bonus(self, speed):
        if speed < 1:
            return (self.collision_bonus(1)
                    if self.use_slow_collision_as_offroad_base else 0)
        else:
            sub_bonus = self.offroad_bonus_above(
                self.wc_non_critical_error_reward)
            return self.offroad_bonus(speed - 1) + sub_bonus

    def collision_bonus(self, speed):
        if speed < 1:
            return 0
        else:
            sub_bonus = self.collision_bonus_above(
                self.wc_non_critical_error_reward)
            return self.collision_bonus(speed - 1) + sub_bonus

    def reward(self, progress_made, speed, car_always_on_pavement,
               *collision_obstacle_speeds):
        r = (self.stopping_reward +
             max(0, progress_made) * self.progress_bonus())
        for cob in collision_obstacle_speeds:
            r += self.collision_bonus(speed + cob)
        if not car_always_on_pavement and speed > 0:
            r += speed * self.offroad_bonus(speed)
        return r

    def __call__(self, s, a, s_prime):
        if s.has_crashed() or s_prime.has_crashed():
            return self.critical_error_reward

        distance = s.car.progress_toward_destination(a)
        car_always_on_pavement = (not (s.is_in_a_ditch()
                                       or s_prime.is_in_a_ditch()))

        def check_for_pedestrian_collision(obs):
            if isinstance(obs, Pedestrian): raise PedestrianHit()

        def check_for_bump_collision(obs=None):
            if obs is None:
                return []
            else:
                return [obs.speed] if isinstance(obs, Bump) else None

        try:
            return self.reward(distance, s.car.speed, car_always_on_pavement,
                               *s.count_obstacle_collisions(
                                   s_prime, check_for_bump_collision,
                                   check_for_pedestrian_collision)[0])
        except PedestrianHit:
            return self.critical_error_reward

    def progress_bonus_below(self, bc_bonus):
        return tf.squeeze(
            tf.random_uniform([self.num_samples], minval=0, maxval=bc_bonus))

    def offroad_bonus_above(self, wc_bonus):
        return self.obstruction_bonus_above(wc_bonus)

    def collision_bonus_above(self, wc_bonus):
        return self.obstruction_bonus_above(wc_bonus)

    def obstruction_bonus_above(self, wc_bonus):
        return tf.squeeze(
            tf.random_uniform([self.num_samples], minval=wc_bonus, maxval=0))


class SituationalReward(_SituationalReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_bonus = None
        self._offroad_bonuses = []
        self._collision_bonuses = []

    def progress_bonus(self):
        if self._progress_bonus is None:
            self._progress_bonus = super().progress_bonus()
        return self._progress_bonus

    def collision_bonus(self, speed):
        if speed < 0:
            return super().collision_bonus(speed)
        else:
            while len(self._collision_bonuses) <= speed:
                self._collision_bonuses.append(super().collision_bonus(
                    len(self._collision_bonuses)))
            return self._collision_bonuses[speed]

    def offroad_bonus(self, speed):
        if speed < 0:
            return super().offroad_bonus(speed)
        else:
            while len(self._offroad_bonuses) <= speed:
                self._offroad_bonuses.append(super().offroad_bonus(
                    len(self._offroad_bonuses)))
            return self._offroad_bonuses[speed]

    def np(self, speed_limit):
        for progress_made, car_always_on_pavement, collision_obstacle_speed in product(
                range(speed_limit + 1), [True, False], range(speed_limit)):
            self.reward(progress_made, car_always_on_pavement,
                        collision_obstacle_speed)
        return (np.array(self._progress_bonus),
                np.array(self._collision_bonuses),
                np.array(self._offroad_bonuses))

    def tf(self, speed_limit):
        for progress_made, car_always_on_pavement, collision_obstacle_speed in product(
                range(speed_limit + 1), [True, False], range(speed_limit)):
            self.reward(progress_made, car_always_on_pavement,
                        collision_obstacle_speed)
        return (tf.stack(self._progress_bonus),
                tf.stack(self._collision_bonuses),
                tf.stack(self._offroad_bonuses))


class ExtremeSituationalReward(SituationalReward):
    def __init__(self, *args, epsilon=1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon


class WcSituationalReward(ExtremeSituationalReward):
    def progress_bonus_below(self, bc_bonus):
        return 0 + self.epsilon

    def obstruction_bonus_above(self, wc_bonus):
        return wc_bonus + self.epsilon


class BcSituationalReward(ExtremeSituationalReward):
    def progress_bonus_below(self, bc_bonus):
        return bc_bonus - self.epsilon

    def obstruction_bonus_above(self, _):
        return 0 - self.epsilon


class ComponentAvgSituationalReward(SituationalReward):
    def progress_bonus_below(self, bc_bonus):
        return bc_bonus / 2.0

    def obstruction_bonus_above(self, wc_bonus):
        return wc_bonus / 2.0


class SampleAvgSituationalReward(SituationalReward):
    def progress_bonus(self):
        return tf.reduce_mean(super().progress_bonus())

    def collision_bonus(self, speed):
        return tf.reduce_mean(super().collision_bonus(speed))

    def offroad_bonus(self, speed):
        return tf.reduce_mean(super().offroad_bonus(speed))


def fixed_ditch_bonus(speed, progress_bonus):
    return -2 * progress_bonus * speed


def critical_reward_for_fixed_ditch_bonus(speed_limit,
                                          progress_bonus,
                                          discount=1.0):
    r = -progress_bonus * (speed_limit + 1)
    if 0 <= discount < 1:
        r /= (1.0 - discount)
    return r - progress_bonus


class DebrisPerceptionReward(SituationalReward):
    def __init__(self, *args, loc=0, precision=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.loc = loc

    def collision_bonus(self, speed):
        if speed < 0:
            return 0
        else:
            while len(self._collision_bonuses) <= speed:
                self._collision_bonuses.append(
                    self._collision_bonus(len(self._collision_bonuses)))
            return self._collision_bonuses[speed]

    def _collision_bonus(self, speed):
        if self.precision is None or np.isinf(self.precision):
            return tf.squeeze(tf.fill([self.num_samples], self.loc))
        else:
            return tf.squeeze(
                tf.random_normal(
                    [self.num_samples],
                    mean=self.loc,
                    stddev=speed / tf.sqrt(self.precision)))

    def offroad_bonus_above(self, wc_bonus):
        return wc_bonus

    def progress_bonus_below(self, bc_bonus):
        return bc_bonus
