#!/usr/bin/env python
import numpy as np
import math
from driving_gridworld.matplotlib import Simulator
from driving_gridworld.matplotlib import Bumps
from driving_gridworld.matplotlib import Crashes
from driving_gridworld.matplotlib import Ditch
from driving_gridworld.matplotlib import Progress
from driving_gridworld.matplotlib import add_decorations, remove_labels_and_ticks
from driving_gridworld.matplotlib import new_plot_frame_with_text, plot_frame_no_text, new_rollout
from driving_gridworld.gridworld import DrivingGridworld
from driving_gridworld.human_ui import observation_to_img, obs_to_rgb
from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.actions import NO_OP
from driving_gridworld.obstacles import Pedestrian, Bump
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')
import os


def ensure_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        return


# Define path where files will be saved:
my_path = os.path.dirname(os.path.realpath(__file__))
dir_name = my_path + '/../tmp'
ensure_dir(dir_name)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']


def new_road(headlight_range=2):
    return Road(
        headlight_range,
        Car(2, 2),
        obstacles=[Bump(0, 2), Pedestrian(1, 1)],
        allowed_obstacle_appearance_columns=[{2}, {1}],
        allow_crashing=True)


def test_still_image_with_no_text():
    game = DrivingGridworld(new_road)
    observation = game.its_showtime()[0]
    img = observation_to_img(observation, obs_to_rgb)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = add_decorations(img, remove_labels_and_ticks(ax))
    ax.imshow(img, aspect=1.8)
    fig.savefig(dir_name + '/img_no_text.pdf')


def test_still_image_with_text():
    game = DrivingGridworld(new_road)
    observation = game.its_showtime()[0]
    img = observation_to_img(observation, obs_to_rgb)
    fig, ax = plt.subplots(figsize=(6, 6))

    reward_function_list = [Progress(), Bumps(), Ditch(), Crashes()]
    info_lists = []
    info_lists.append([f.new_info() for f in reward_function_list])
    frame, ax_texts = new_plot_frame_with_text(
        img, 0, *info_lists[0], fig=fig, ax=ax)[:2]
    fig.savefig(dir_name + '/img_with_text.pdf')


def test_video_with_text():
    frames, fig, ax_list, actions, rollout_info_lists = new_rollout(
        Simulator(lambda x: NO_OP, DrivingGridworld(new_road)),
        plotting_function=new_plot_frame_with_text,
        reward_function_list=[Progress(),
                              Bumps(), Ditch(),
                              Crashes()],
        num_steps=10)
    ani = animation.ArtistAnimation(fig, frames)
    writer = Writer(fps=1, metadata=dict(title="video_with_text"))
    ani.save(dir_name + '/video_with_text.mp4', writer=writer)


def test_video_with_no_text():  # Should maybe pass the policy as an argument?
    frames, fig, ax_list, actions, rollout_info_lists = new_rollout(
        Simulator(lambda x: NO_OP, DrivingGridworld(new_road)),
        plotting_function=plot_frame_no_text,
        reward_function_list=[Progress(),
                              Bumps(), Ditch(),
                              Crashes()],
        num_steps=10)
    ani = animation.ArtistAnimation(fig, frames)
    writer = Writer(fps=1, metadata=dict(title="video_no_text"))
    ani.save(dir_name + '/video_no_text.mp4', writer=writer)


def test_video_multiple_agents_with_text():
    pass


def test_video_multiple_agents_with_text():
    pass


if __name__ == '__main__':
    test_still_image_with_no_text()
    test_still_image_with_text()
    test_video_with_text()
    test_video_with_no_text()
    
