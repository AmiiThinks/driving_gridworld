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


def plot_frame_with_text(img, r, discounted_return, a):
    dim = img.shape
    white_matrix = np.ones(dim)
    extended_img = np.concatenate((img, white_matrix), axis=1)

    t1 = 'Action: ' + str(action_to_string[a])
    t2 = 'Reward: {:0.2f}'.format(r)
    t3 = 'Return: {:0.2f}'.format(discounted_return)

    frame, fig, ax = plot_frame(extended_img, animated=True)

    column = img.shape[1] - 0.4
    plt.text(column, 0, t1)
    plt.text(column, 1, t2)
    plt.text(column, 2, t3)

    return frame, fig, ax
