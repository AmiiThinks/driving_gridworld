import matplotlib.pyplot as plt


def plot_frame(img, fig=None, ax=None, animated=False, show_grid=False):
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

    return plt.imshow(img, animated=animated), fig, ax
