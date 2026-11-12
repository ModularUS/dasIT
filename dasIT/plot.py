'''
Licensed under the Apache License, Version 2.0, see the LICENSE file of the dasIT repository.

One image on axes in millimetres, with the array centre at x = 0 and depth running down.
'''

import numpy as np
from matplotlib import pyplot as plt


def show_image(image_db, grid_x, grid_z, ax=None, title=None, depth_range=None):
    """Draw a log-compressed image [depth, lateral] (dB, as from dasIT.signal.log_compress).
    grid_x, grid_z: pixel coordinates [m]. depth_range = (from, to) [m] shows only that part; nothing else is cut and
    the inputs are not changed. Returns the axes."""
    grid_x, grid_z = np.asarray(grid_x, dtype=float), np.asarray(grid_z, dtype=float)
    rows = np.arange(grid_z.size) if depth_range is None else np.flatnonzero((grid_z >= depth_range[0]) & (grid_z <= depth_range[1]))
    dx = (grid_x[1] - grid_x[0]) / 2 if grid_x.size > 1 else 0.0
    dz = (grid_z[1] - grid_z[0]) / 2 if grid_z.size > 1 else 0.0
    extent = [1e3 * (grid_x[0] - dx), 1e3 * (grid_x[-1] + dx), 1e3 * (grid_z[rows[-1]] + dz), 1e3 * (grid_z[rows[0]] - dz)]
    ax = ax or plt.subplots(figsize=(5, 6))[1]
    picture = ax.imshow(np.asarray(image_db)[rows], cmap='gray', extent=extent, aspect='equal', interpolation='nearest')
    ax.set_xlabel('Lateral [mm]')
    ax.set_ylabel('Axial [mm]')
    if title:
        ax.set_title(title)
    ax.figure.colorbar(picture, ax=ax, label='dB')
    return ax
