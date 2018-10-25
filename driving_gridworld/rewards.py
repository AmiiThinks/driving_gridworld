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
                 epsilon=0,
                 critical_error_reward=-10.0,
                 bc_unobstructed_progress_reward=1.0,
                 num_samples=1):
        self.num_samples = num_samples
        self.wc_non_critical_error_reward = wc_non_critical_error_reward
        self.stopping_reward = stopping_reward
        self.epsilon = epsilon
        self.critical_error_reward = critical_error_reward
        self.bc_unobstructed_progress_reward = bc_unobstructed_progress_reward

    def progress_bonus(self):
        return self.progress_bonus_between(
            self.stopping_reward, self.bc_unobstructed_progress_reward)

    def unobstructed_reward(self, progress_made):
        if progress_made < 1:
            return self.stopping_reward
        else:
            return (self.stopping_reward + progress_made *
                    (self.progress_bonus() + self.epsilon))

    def offroad_bonus(self, speed):
        if speed < 0:
            return self.stopping_reward
        else:
            sub_bonus = self.offroad_bonus_between(
                self.wc_non_critical_error_reward, self.stopping_reward)
            return self.offroad_bonus(speed - 1) + sub_bonus - self.epsilon

    def offroad_reward(self, progress_made, speed):
        return (self.unobstructed_reward(progress_made) +
                self.offroad_bonus(speed))

    def collision_bonus(self, speed):
        if speed < 0:
            return self.stopping_reward
        else:
            sub_bonus = self.collision_bonus_between(
                self.wc_non_critical_error_reward, self.stopping_reward)
            return self.collision_bonus(speed - 1) + sub_bonus - self.epsilon

    def pavement_collision_reward(self, progress_made, speed,
                                  collision_obstacle_speed):
        return (self.unobstructed_reward(progress_made) +
                self.collision_bonus(speed + collision_obstacle_speed))

    def offroad_collision_reward(self, progress_made, speed,
                                 collision_obstacle_speed):
        return (self.pavement_collision_reward(progress_made, speed,
                                               collision_obstacle_speed) +
                self.offroad_bonus(speed))

    def reward(self,
               progress_made,
               speed,
               car_always_on_pavement,
               collision_obstacle_speed=None):
        if collision_obstacle_speed is None:
            if car_always_on_pavement:
                return self.unobstructed_reward(progress_made)
            else:
                return self.offroad_reward(progress_made, speed)
        else:
            if car_always_on_pavement:
                return self.pavement_collision_reward(progress_made, speed,
                                                      collision_obstacle_speed)
            else:
                return self.offroad_collision_reward(progress_made, speed,
                                                     collision_obstacle_speed)

    def __call__(self, s, a, s_prime):
        if s.has_crashed() or s_prime.has_crashed():
            return self.critical_error_reward

        distance = s.car.progress_toward_destination(a)
        car_always_on_pavement = (not (s.is_in_a_ditch()
                                       or s_prime.is_in_a_ditch()))
        reward_without_collision = self.reward(distance, s.car.speed,
                                               car_always_on_pavement)

        def check_for_pedestrian_collision(obs):
            if isinstance(obs, Pedestrian): raise PedestrianHit()

        def check_for_bump_collision(obs):
            return ((self.reward(distance, s.car.speed, car_always_on_pavement,
                                 obs.speed) - reward_without_collision)
                    if isinstance(obs, Bump) else None)

        try:
            rewards_collided_obstacles = s.count_obstacle_collisions(
                s_prime, check_for_pedestrian_collision,
                check_for_bump_collision)
        except PedestrianHit:
            return self.critical_error_reward

        return sum(rewards_collided_obstacles) + reward_without_collision

    def progress_bonus_between(self, min_reward, max_reward):
        return self._random_uniform(min_reward, max_reward)

    def offroad_bonus_between(self, min_r, max_r):
        return self._random_uniform(min_r, max_r)

    def collision_bonus_between(self, min_r, max_r):
        return self._random_uniform(min_r, max_r)

    def _random_uniform(self, minval=0, maxval=1):
        return tf.squeeze(
            tf.random_uniform(
                [self.num_samples], minval=minval, maxval=maxval))

    def _min(self, *args):
        return tf.reduce_min(args, axis=0)


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


class WcSituationalReward(SituationalReward):
    def __init__(self, *args, epsilon=1e-10, **kwargs):
        super().__init__(*args, epsilon=epsilon, **kwargs)

    def progress_bonus_between(self, min_reward, _):
        return min_reward

    def offroad_bonus_between(self, min_r, max_r):
        return min_r

    def collision_bonus_between(self, min_r, max_r):
        return min_r


class BcSituationalReward(SituationalReward):
    def __init__(self, *args, epsilon=1e-10, **kwargs):
        super().__init__(*args, epsilon=epsilon, **kwargs)

    def progress_bonus_between(self, min_r, max_r):
        return max_r

    def offroad_bonus_between(self, min_r, max_r):
        return max_r

    def collision_bonus_between(self, min_r, max_r):
        return max_r


class ComponentAvgSituationalReward(SituationalReward):
    def progress_bonus_between(self, min_r, max_r):
        return (max_r + min_r) / 2.0

    def offroad_bonus_between(self, min_r, max_r):
        return (max_r + min_r) / 2.0

    def collision_bonus_between(self, min_r, max_r):
        return (max_r + min_r) / 2.0


class SampleAvgSituationalReward(SituationalReward):
    def progress_bonus(self):
        return tf.reduce_mean(super().progress_bonus())

    def collision_bonus(self, speed):
        return tf.reduce_mean(super().collision_bonus(speed))

    def offroad_bonus(self, speed):
        return tf.reduce_mean(super().offroad_bonus(speed))
