import matplotlib.pyplot as plt
import numpy as np


def plot_model_and_grav(blocks, grav, x_vals, **kwds):
    # Assuming the setup variables x_vals, grav, and blocks are defined.
    x_min, x_max = 0, 200
    y_min, y_max = -100, 0

    vmin = kwds.get("v_min", 1)
    vmax = kwds.get("v_max", 5)

    grav_min = kwds.get("grav_min", grav.min())
    grav_max = kwds.get("grav_max", grav.max())

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    try:
        ax1.plot(x_vals, grav.detach().numpy(), '.-')
    except AttributeError:
        ax1.plot(x_vals, grav, '.-')

    if grav_min:
        ax1.set_ylim([grav_min, grav_max])

    ax1.set_ylabel("Gravity anomaly $\Delta g_z$ [m/s$^2$]")

    # Add a transparent gray rectangle to the left side of the plot
    # rect = patches.Rectangle((0, grav_min), width=50, height=grav_max-grav_min, facecolor='gray', alpha=0.2)
    # ax1.add_patch(rect)
    # Add a transparent gray rectangle to the left side of the plot
    # rect = patches.Rectangle((150, grav_min), width=50, height=grav_max-grav_min, facecolor='gray', alpha=0.2)
    # ax1.add_patch(rect)

    ax2 = fig.add_subplot(212, sharex=ax1)

    # Correctly define your grid edges
    y_edges = np.linspace(y_min, y_max, blocks.shape[2] + 1)
    x_edges = np.linspace(x_min, x_max, blocks.shape[0] + 1)  # This should be corrected if it was incorrect

    # Making sure x_edges and y_edges are correctly sized
    assert len(x_edges) == blocks.shape[0] + 1, "x_edges does not match expected length"
    assert len(y_edges) == blocks.shape[2] + 1, "y_edges does not match expected length"

    # Ensure blocks[:, 5, :].T is correctly shaped relative to x_edges and y_edges
    c = ax2.pcolor(x_edges, y_edges, blocks[:, 5, :].T, cmap='viridis', vmin=vmin, vmax=vmax)
    # c = ax2.pcolor(blocks[:, 5, :].T, cmap='RdYlBu_r') # , vmin=1, vmax=2)

    # plot input data if given in kwds:
    if "input_data" in kwds.keys():
        plt.scatter(kwds["input_data"]['X'], kwds["input_data"]['Z'], color='#CCCCCC', edgecolors='black', s=100)

    # remove ticks from upper plot
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("z [m]")

    # Add a transparent gray rectangle to the left side of the plot
    # rect = patches.Rectangle((0, -100), width=50, height=100, facecolor='white', alpha=0.2)
    # ax2.add_patch(rect)
    # rect = patches.Rectangle((150, -100), width=50, height=100, facecolor='white', alpha=0.2)
    # ax2.add_patch(rect)

    # Adjust the subplot layout to prevent the subplots from overlapping
    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(left=0.1, right=.95, bottom=0.1, top=0.95)

    plt.close()

    return fig
