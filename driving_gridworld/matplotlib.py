import matplotlib.pyplot as plt
import numpy as np
from driving_gridworld.actions import ACTION_NAMES


def plot_frame_with_text(
        img,
        reward,
        discounted_return,
        action,
        fig=None,
        ax=None,
        animated=False,
        show_grid=False):
    white_matrix = np.ones(img.shape)
    extended_img = np.concatenate((img, white_matrix), axis=1)

    text_list = [
      'Action: {}'.format(ACTION_NAMES[action]),
      'Reward: {:0.2f}'.format(reward),
      'Return: {:0.2f}'.format(discounted_return)
    ]

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    ax.grid(show_grid)

    # Remove ticks and tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    column = img.shape[1] - 0.4
    ax_texts = [
        ax.annotate(t, (column, i)) for i, t in enumerate(text_list)
    ]

    return ax.imshow(extended_img, animated=animated), ax_texts, fig, ax
