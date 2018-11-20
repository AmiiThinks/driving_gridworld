#!/usr/bin/env python
import numpy as np
from driving_gridworld.matplotlib import RewardInfo
from driving_gridworld.matplotlib import RewardFunction
from driving_gridworld.matplotlib import Bumps
from driving_gridworld.matplotlib import Crashes
from driving_gridworld.matplotlib import Ditch
from driving_gridworld.matplotlib import Progress
from driving_gridworld.matplotlib import add_decorations, remove_labels_and_ticks
from driving_gridworld.gridworld import DrivingGridworld
from driving_gridworld.matplotlib import new_plot_frame_with_text
from driving_gridworld.human_ui import observation_to_img, obs_to_rgb
from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Pedestrian, Bump
import matplotlib.pyplot as plt
import os


def new_road(headlight_range=2):
    return Road(
        headlight_range,
        Car(2, 2),
        obstacles=[Bump(0, 2), Pedestrian(1, 1)],
        allowed_obstacle_appearance_columns=[{2}, {1}],
        allow_crashing=True)


def ensure_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        return


def test_still_image_with_no_text():
    game = DrivingGridworld(new_road)
    observation = game.its_showtime()[0]
    img = observation_to_img(observation, obs_to_rgb)

    fig, ax = plt.subplots(figsize=(3, 10))
    ax = add_decorations(img, remove_labels_and_ticks(ax))
    ax.imshow(img, aspect=1.5)
    # plt.show()
    my_path = os.path.dirname(os.path.realpath(__file__))
    dir_name = my_path + '/../tmp'
    ensure_dir(dir_name)
    fig.savefig(dir_name + '/img_no_text.pdf')


def test_still_image_with_text():
    game = DrivingGridworld(new_road)
    observation = game.its_showtime()[0]
    img = observation_to_img(observation, obs_to_rgb)
    reward_function_list = [Progress(), Bumps(), Ditch(), Crashes()]
    info_lists = []
    frames = [[]]
    info_lists.append([f.new_info() for f in reward_function_list])
    fig, ax = plt.subplots(figsize=(3, 15))
    frame, ax_texts = new_plot_frame_with_text(
        img, 0, *info_lists[0], fig=fig, ax=ax)[:2]
    frames[0] += [frame] + ax_texts

    ax = add_decorations(img, remove_labels_and_ticks(ax))
    ax.imshow(img, aspect=1.5)
    plt.show()
    my_path = os.path.dirname(os.path.realpath(__file__))
    dir_name = my_path + '/../tmp'
    ensure_dir(dir_name)
    fig.savefig(dir_name + '/img_with_text.pdf')


if __name__ == '__main__':
    test_still_image_with_no_text()
