#!/usr/bin/env python
"""A simple driving simulator.

Command-line usage: `road.py`.

Keys: left, right - move. up, down - speed up or down, respectively. q - quit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.enable_eager_execution()

import fire
import numpy as np
import pickle

from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Pedestrian, Bump
from driving_gridworld.rewards import ComponentAvgSituationalReward


def save(data, path):
    with open(path + ".pkl", 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(headlight_range=3,
         num_steps=100,
         ui=False,
         recording_path=None,
         discount=0.99,
         allow_crashing=True):
    np.random.seed(42)

    headlight_range = int(headlight_range)
    num_steps = int(num_steps)
    ui = bool(ui)

    def new_road():
        return Road(
            headlight_range,
            Car(2, 0),
            obstacles=[
                Bump(-1, -1, prob_of_appearing=0.16),
                Pedestrian(-1, -1, speed=1, prob_of_appearing=0.13)
            ],
            allowed_obstacle_appearance_columns=[{2}, {1}],
            allow_crashing=allow_crashing)

    speed_limit = headlight_range + 1
    wc_non_critical_error_reward = -1.0 / speed_limit

    reward_function = ComponentAvgSituationalReward(
        wc_non_critical_error_reward=wc_non_critical_error_reward,
        stopping_reward=0,
        critical_error_reward=((speed_limit * speed_limit + speed_limit) *
                               wc_non_critical_error_reward - 1000.0),
        bc_unobstructed_progress_reward=1.0 / speed_limit)

    if ui:
        from driving_gridworld.human_ui import UiRecordingDrivingGridworld

        game = UiRecordingDrivingGridworld(
            new_road,
            discount=discount,
            reward_function=lambda s, a, sp: tf.convert_to_tensor(reward_function(s, a, sp)).numpy()
        )
        game.ui_play()
    else:
        from driving_gridworld.gridworld import RecordingDrivingGridworld

        game = RecordingDrivingGridworld(
            new_road,
            discount=discount,
            reward_function=lambda s, a, sp: tf.convert_to_tensor(reward_function(s, a, sp)).numpy()
        )

        observation, _, __ = game.its_showtime()

        rl_return = 0.0
        discount_product = 1.0
        for _ in range(num_steps):
            a = np.random.randint(0, 4)
            observation, reward, discount = game.play(a)
            discount_product *= discount
            rl_return += reward * discount_product

            # print(
            #     (
            #         a,
            #         reward, discount,
            #         rl_return
            #     )
            # )
            # board, speed = game.observation_to_key(observation)
            # print(board)
            # print(speed)
            # print('')
        print(
            'Final return for uniform random policy after {} steps: {}'.format(
                num_steps, rl_return))

    if recording_path is not None and len(recording_path) > 0:
        print('Saving recording to "{}".'.format(recording_path))
        save(game.recorded(), recording_path)


if __name__ == '__main__':
    fire.Fire(main)
