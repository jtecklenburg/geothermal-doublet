import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_with_color_gradient(x, y, z, zmin, zmax, cmap='viridis', linewidth=2, ax=None):
    """
    Plot a line with a color gradient based on values in z.

    Parameters:
    x (array-like): x coordinates of the points.
    y (array-like): y coordinates of the points.
    z (array-like): values for the color gradient.
    zmin (float): minimum value for the color gradient.
    zmax (float): maximum value for the color gradient.
    cmap (str): colormap to use for the gradient.
    linewidth (float): width of the line.
    ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses current axes.
    """
    if ax is None:
        ax = plt.gca()

    # Normalize the values for the color gradient
    norm = Normalize(vmin=zmin, vmax=zmax)
    color_map = plt.get_cmap(cmap)

    # Create the segments for the color gradient
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the LineCollection
    lc = LineCollection(segments, cmap=color_map, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(linewidth)

    # Add the LineCollection to the axes
    ax.add_collection(lc)


def finalize_plot():
    """
    Finalize the plot by adding labels, title, and colorbar, and show it.
    """
    ax = plt.gca()
    ax.autoscale()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # ax.set_title('Line plot with color gradient depending on Z')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Age (a)')
    plt.show()


if __name__ == "__main__":
    # Example data and plotting
    plt.figure()

    # First dataset
    x1 = np.linspace(0, 10, 100)
    y1 = np.sin(x1)
    z1 = y1  # The value Z, which determines the color gradient
    plot_with_color_gradient(x1, y1, z1, zmin=-1, zmax=1)

    # Second dataset
    x2 = np.linspace(0, 10, 100)
    y2 = np.cos(x2)
    z2 = y2  # The value Z, which determines the color gradient
    plot_with_color_gradient(x2, y2, z2, zmin=-1, zmax=1)

    # Finalize the plot
    finalize_plot()
