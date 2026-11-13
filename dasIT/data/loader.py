'''
Licensed under the Apache License, Version 2.0, see the LICENSE file of the dasIT repository.

Reading recorded channel data.
'''

import h5py
import numpy as np


def load_rf(path, dtype=np.float64):
    """RF of an h5 file with the datasets frameNNNN/shotNNNN, each [samples, channels].
    Returns [samples, channels, shots, frames]: shots (the transmits of one image) and frames keep their own axis."""
    with h5py.File(path, 'r') as file:
        frames = sorted(file.keys())
        shots = sorted(file[frames[0]].keys())
        data = np.stack([np.stack([file[f'{f}/{s}'][:] for s in shots], axis=-1) for f in frames], axis=-1)
    return data.astype(dtype)
