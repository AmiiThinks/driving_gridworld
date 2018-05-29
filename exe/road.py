"""A simple driving simulator.

Command-line usage: `road.py`.

Keys: left, right - move. up, down - speed up or down, respectively. q - quit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import curses
except:
    print('Warning: Not importing `curses`.')
try:
    import fire
except:
    print('Warning: Not importing `fire`.')

import numpy as np
from road_gridworld import *


def main(
    num_rows=5,
    num_bumps=3,
    num_pedestrians=3,
    speed=1,
    speed_limit=3,
    num_steps=100,
    ui=False
):
    np.random.seed(42)

    num_rows = int(num_rows)
    num_bumps = int(num_bumps)
    num_pedestrians = int(num_pedestrians)
    speed = int(speed)
    speed_limit = int(speed_limit)
    num_steps = int(num_steps)

    game = RoadPycolabEnv(
        num_rows,
        num_bumps,
        num_pedestrians,
        speed,
        speed_limit)

    if ui:
        game.ui_play()
    else:
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


if __name__ == '__main__':
    fire.Fire(main)
