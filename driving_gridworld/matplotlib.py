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
    dim = img.shape
    white_matrix = np.ones(dim)
    extended_img = np.concatenate((img, white_matrix), axis=1)

    t1 = 'Action: ' + ACTION_NAMES[action]
    t2 = 'Reward: {:0.2f}'.format(reward)
    t3 = 'Return: {:0.2f}'.format(discounted_return)

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    plt.grid(show_grid)

    # Remove ticks and tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    column = img.shape[1] - 0.4
    plt.text(column, 0, t1)
    plt.text(column, 1, t2)
    plt.text(column, 2, t3)

    return plt.imshow(extended_img, animated=animated), fig, ax
