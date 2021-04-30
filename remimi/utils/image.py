import matplotlib.pyplot as plt

def show_image_blocking(image, **imshow_arg):
    fig = plt.figure(frameon=False)
    # fig.set_size_inches(image.shape[1], image.shape[0])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto', **imshow_arg)
    plt.show()