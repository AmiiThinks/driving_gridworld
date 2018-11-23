import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
from driving_gridworld.actions import ACTION_NAMES
from driving_gridworld.rollout import Rollout
from driving_gridworld.human_ui import observation_to_img, obs_to_rgb
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.actions import NO_OP


class RewardInfo(object):
    def __init__(self, name, string_format, discount=1.0):
        self.name = name
        self.discount = discount
        self.g = 0
        self._t = 0
        self._string_format = string_format
        self.r = 0

    def next(self, reward_value):
        if self._t > 0:
            if self.discount < 1.0:
                self.g += self.discount**(self._t - 1) * self.r
            else:
                self.g += self.r
        self.r = reward_value
        self._t += 1
        return self

    def reward_to_s(self):
        return self._string_format.format(self.r)

    def return_to_s(self):
        return self._string_format.format(self.g)


class RewardFunction(object):
    def __init__(self, name, discount=1.0):
        self.name = name
        self.discount = discount

    def new_info(self):
        return RewardInfo(
            self.name, self.string_format, discount=self.discount)


class Bumps(RewardFunction):
    def __init__(self):
        super().__init__('Bumps', 1.0)

    def __call__(self, s, a, s_p):
        return s.count_obstacle_collisions(
            s_p, lambda obs: 1 if isinstance(obs, Bump) else None)[0]

    @property
    def string_format(self):
        return '{:d}'


class Crashes(RewardFunction):
    def __init__(self):
        super().__init__('Crashes', 1.0)

    def __call__(self, s, a, s_p):
        return s.count_obstacle_collisions(
            s_p, lambda obs: 1 if isinstance(obs, Pedestrian) else None)[0]

    @property
    def string_format(self):
        return '{:d}'


class Ditch(RewardFunction):
    def __init__(self):
        super().__init__('Ditch', 1.0)

    def __call__(self, s, a, s_p):
        return int(s.is_in_a_ditch() or s_p.is_in_a_ditch()) * s.car.speed

    @property
    def string_format(self):
        return '{:d}'


class Progress(RewardFunction):
    def __init__(self):
        super().__init__('Progress', 1.0)

    def __call__(self, s, a, s_p):
        return s.car.progress_toward_destination(a)

    @property
    def string_format(self):
        return '{:d}'


def remove_labels_and_ticks(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.grid(False)
    ax.axis('off')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    return ax


def add_decorations(img, ax=None):
    if ax is None:
        ax = plt.gca()

    incr = 0.55
    y = -0.60
    for i in range(2 * img.shape[0]):
        ax.add_patch(
            mpl.patches.Rectangle(
                (2.47, y), 0.03, 0.33, 0, color='yellow', alpha=0.8))
        y += incr

    direction_offsets = np.array([(-0.5, -0.5), (0, -0.5 - 0.5 / 3), (0.5,
                                                                      -0.5)])
    for i in range(img.shape[0] - 1):
        ax.add_patch(
            mpl.patches.Polygon(
                np.array([6, i + 1]) + direction_offsets,
                closed=True,
                alpha=0.8,
                facecolor='grey'))
    return ax


def new_plot_frame_with_text(img,
                             action,
                             *reward_info_list,
                             fig=None,
                             ax=None,
                             animated=False,
                             show_grid=False):

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    ax = add_decorations(img, remove_labels_and_ticks(ax))
    num_text_columns = 5
    white_matrix = np.ones([img.shape[0], num_text_columns, img.shape[2]])
    extended_img = np.concatenate((img, white_matrix), axis=1)

    text_list = [ACTION_NAMES[action]]
    for info in reward_info_list:
        text_list.append('{:8s} {:>5s} + {:>1s}'.format(
            info.name, info.return_to_s(), info.reward_to_s()))

    column = img.shape[1] - 0.1
    font = mpl.font_manager.FontProperties()
    font.set_family('monospace')
    ax_texts = [
        ax.text(
            column,
            math.ceil(img.shape[0] / 2),
            '\n\n'.join(text_list[0:]),
            horizontalalignment='left',
            fontproperties=font)
    ]

    return ax.imshow(
        extended_img, animated=animated, aspect=1.8), ax_texts, fig, ax


def plot_frame_no_text(img,
                       action,
                       *reward_info_list,
                       fig=None,
                       ax=None,
                       animated=False,
                       show_grid=False):
    num_text_columns = 0
    white_matrix = np.ones([img.shape[0], num_text_columns, img.shape[2]])
    extended_img = np.concatenate((img, white_matrix), axis=1)

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    remove_labels_and_ticks(ax)
    add_decorations(img, ax)

    return ax.imshow(extended_img, animated=animated, aspect=1.8), [], fig, ax


def new_rollout(*simulators,
                plotting_function=plot_frame_no_text,
                reward_function_list=[],
                num_steps=100,
                fig=None,
                ax_list=None):
    if fig is None or ax_list is None:
        fig, ax_list = plt.subplots(
            len(simulators), figsize=(6, 6), squeeze=False)
        ax_list = ax_list.reshape([len(simulators)])

    info_lists = []
    frames = [[]]
    for i, sim in enumerate(simulators):
        observation, d = sim.start()
        img = observation_to_img(observation, obs_to_rgb)
        info_lists.append([f.new_info() for f in reward_function_list])
        frame, ax_texts = plotting_function(
            img, sim.a, *info_lists[i], fig=fig, ax=ax_list[i])[:2]

        frames[0] += [frame] + ax_texts

    actions = [[] for _ in simulators]
    for t in range(num_steps):
        frames.append([])
        for i, sim in enumerate(simulators):
            a, observation, _ = sim.step()
            actions[i].append(a)

            for j, info in enumerate(info_lists[i]):
                info.next(reward_function_list[j](*sim.sas()))

            frame, ax_texts = plotting_function(
                observation_to_img(observation, obs_to_rgb),
                a,
                *info_lists[i],
                fig=fig,
                ax=ax_list[i])[:2]
            frames[-1] += [frame] + ax_texts
    return frames, fig, ax_list, actions, info_lists


class Simulator(object):
    def __init__(self, policy, game):
        self.policy = policy
        self.game = game

    def start(self):
        self.prev_state = self.game.road.copy()
        self.observation, _, d = self.game.its_showtime()
        self.a = NO_OP
        self.d = 1
        return self.observation, d

    def step(self):
        if self.d > 0:
            self.prev_state = self.game.road.copy()
            self.a = self.policy(self.game.road)
            self.observation, _, self.d = self.game.play(self.a)
        return self.a, self.observation, self.d

    def sas(self):
        return self.prev_state, self.a, self.game.road
