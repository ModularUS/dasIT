'''
Licensed under the Apache License, Version 2.0, see the LICENSE file of the dasIT repository.

Delay-and-sum beamforming from explicit inputs in SI units.

    bf = Beamformer(element_x, grid_x, grid_z, fs, c, tx=plane_wave(0))
    image = bf(rf)                     rf: [samples, channels] or [samples, channels, frames], real or analytic

For one receive element and every pixel (x, z) there are two numbers:

    arrival sample   s = ( tx(x, z) + distance(pixel, element) / c - t0 ) * fs          Beamformer._arrival
    weight           w = window( |x - x_element| / half_aperture(z) )                   Beamformer._weight

The image is the sum over the elements of  w * rf[s, element].  That is all of delay-and-sum. Written out for all
elements it is a list of triplets (pixel, sample of a channel, weight), which `_taps` produces element by element
and only for the pixels inside the aperture. The two engines differ in what they do with the triplets:

    engine "matrix"   keep all of them in one sparse matrix, image = matrix @ rf.
                      Fast, and any number of frames in one product. Memory grows with pixels x aperture.
    engine "loop"     add them to the image element by element and forget them. A few megabytes, for large grids.

Both engines use the same triplets and give the same image.
'''

import numpy as np
from scipy import sparse


def _box(u):
    return (u <= 1.0).astype(float)


def _hann(u):
    return np.where(u < 1.0, 0.5 + 0.5 * np.cos(np.pi * np.minimum(u, 1.0)), 0.0)


WINDOWS = {'box': _box, 'hann': _hann}      # weight against u = distance from the pixel / half aperture
INTERPOLATIONS = ('nearest', 'linear')
ENGINES = ('matrix', 'loop')


class Beamformer():
    def __init__(self, element_x, grid_x, grid_z, fs, c, tx, t0=0.0, f_number=1.5, window='hann', interp='linear',
                 engine='matrix', normalise=False, half_aperture=None, dtype=np.float64):
        """
        element_x      x of the receive elements [m], one per data channel (dasIT.geometry.linear_array)
        grid_x, grid_z pixel coordinates [m], any spacing
        fs, c          sampling rate [Hz], speed of sound [m/s]
        tx             transmit delay tx(x, z, c) -> s (dasIT.geometry.plane_wave)
        t0             time of sample 0 on the clock of tx [s]
        f_number       receive aperture: its width at depth z is z / f_number, at least one element spacing
        window         'box' or 'hann', the weight across the aperture
        interp         'nearest' sample, or 'linear' between the two samples around the arrival time
        engine         'matrix' or 'loop'
        normalise      divide every pixel by the sum of its weights: brightness independent of the aperture size
        half_aperture  half width of the aperture per depth [m], replaces the f_number rule when given
        dtype          np.float64, or np.float32 for less memory
        """
        for value, choices, name in ((window, WINDOWS, 'window'), (interp, INTERPOLATIONS, 'interp'), (engine, ENGINES, 'engine')):
            if value not in choices:
                raise ValueError(f"Unknown {name} '{value}'. Choose one of {sorted(choices)}.")
        self.element_x = np.asarray(element_x, dtype=float)
        self.grid_x = np.asarray(grid_x, dtype=float)
        self.grid_z = np.asarray(grid_z, dtype=float)
        self.fs, self.c, self.t0 = float(fs), float(c), float(t0)
        self.window, self.interp, self.engine, self.normalise, self.dtype = window, interp, engine, normalise, dtype
        self.shape = (self.grid_z.size, self.grid_x.size)

        self._tx = np.ascontiguousarray(np.broadcast_to(tx(self.grid_x[None, :], self.grid_z[:, None], self.c), self.shape))
        if half_aperture is None:
            spacing = np.median(np.diff(self.element_x)) if self.element_x.size > 1 else 0.0
            half_aperture = np.maximum(self.grid_z / (2 * f_number), spacing / 2)
        self._half_aperture = np.maximum(np.broadcast_to(half_aperture, self.grid_z.shape), 1e-12)
        self._matrices = {}

    # ---- the physics: one receive element, the pixels at (x, z)
    def _arrival(self, element, x, z, tx):
        """Sample (fractional) at which the echo from (x, z) arrives on this element."""
        return (tx + np.sqrt(z ** 2 + (x - self.element_x[element]) ** 2) / self.c - self.t0) * self.fs

    def _weight(self, element):
        """Weight of this element for every pixel, [depth, lateral]."""
        return WINDOWS[self.window](np.abs(self.grid_x[None, :] - self.element_x[element]) / self._half_aperture[:, None])

    def delays(self, element):
        """Arrival sample (fractional) of the echo of every pixel on this element, [depth, lateral]."""
        return self._arrival(element, self.grid_x[None, :], self.grid_z[:, None], self._tx)

    def weights(self, element):
        """Weight of this element for every pixel, [depth, lateral]. 0 outside the aperture."""
        return self._weight(element)

    # ---- the triplets of one element: (pixel, sample, weight), only where the weight is not zero
    def _taps(self, element, n_samples):
        """[(pixel, sample, weight)]: one entry for 'nearest', two for 'linear' (the samples before and after the
        arrival). pixel = depth index * lateral size + lateral index. Arrivals outside the record are left out."""
        weight = self._weight(element).ravel()
        pixel = np.flatnonzero(weight)                                   # the pixels that have this element in their aperture
        iz, ix = np.divmod(pixel, self.shape[1])
        s = self._arrival(element, self.grid_x[ix], self.grid_z[iz], self._tx.ravel()[pixel])
        weight = weight[pixel]
        if self.interp == 'nearest':
            first = np.rint(s).astype(np.int64)
            parts = [(first, weight)]
        else:
            first = np.floor(s).astype(np.int64)
            fraction = s - first
            parts = [(first, weight * (1 - fraction)), (first + 1, weight * fraction)]
        inside = (first >= 0) & (first + len(parts) - 1 <= n_samples - 1)
        pixel = pixel[inside].astype(np.int32)
        return [(pixel, sample[inside].astype(np.int32), w[inside].astype(self.dtype)) for sample, w in parts]

    # ---- engine "matrix"
    def matrix(self, n_samples):
        """The beamformer as a sparse matrix [pixels, samples * channels] for records of n_samples. Built once."""
        if n_samples not in self._matrices:
            n_channels = self.element_x.size
            n_parts = 1 if self.interp == 'nearest' else 2
            room = n_parts * sum(np.count_nonzero(self._weight(e)) for e in range(n_channels))     # upper bound, no copies later
            rows, cols, vals = np.empty(room, np.int32), np.empty(room, np.int32), np.empty(room, self.dtype)
            n = 0
            for element in range(n_channels):
                for pixel, sample, weight in self._taps(element, n_samples):
                    rows[n:n + pixel.size], vals[n:n + pixel.size] = pixel, weight
                    cols[n:n + pixel.size] = sample * n_channels + element
                    n += pixel.size
            m = sparse.csr_matrix((vals[:n], (rows[:n], cols[:n])), shape=(self.shape[0] * self.shape[1], n_samples * n_channels))
            if self.normalise:
                total = np.asarray(m.sum(axis=1)).ravel()
                scale = np.divide(1.0, total, out=np.zeros_like(total), where=total != 0)
                m.data *= np.repeat(scale, np.diff(m.indptr)).astype(self.dtype)
            self._matrices[n_samples] = m
        return self._matrices[n_samples]

    # ---- engine "loop"
    def _loop(self, rf):
        n_pixels = self.shape[0] * self.shape[1]
        out = np.zeros((n_pixels,) + rf.shape[2:], dtype=np.result_type(rf.dtype, self.dtype))
        total = np.zeros(n_pixels, dtype=self.dtype)
        for element in range(self.element_x.size):
            for pixel, sample, weight in self._taps(element, rf.shape[0]):          # a pixel occurs once per entry
                out[pixel] += rf[sample, element] * (weight if rf.ndim == 2 else weight[:, None])
                total[pixel] += weight
        if self.normalise:
            scale = np.divide(1.0, total, out=np.zeros_like(total), where=total != 0)
            out *= scale if rf.ndim == 2 else scale[:, None]
        return out.reshape(self.shape + rf.shape[2:])

    def __call__(self, rf):
        """Beamform rf [samples, channels] or [samples, channels, frames] -> [depth, lateral] or [depth, lateral, frames]."""
        rf = np.asarray(rf)
        if rf.ndim not in (2, 3) or rf.shape[1] != self.element_x.size:
            raise ValueError(f"rf must be [samples, {self.element_x.size} channels] or [samples, channels, frames], got {rf.shape}")
        if self.engine == 'loop':
            return self._loop(rf)
        image = self.matrix(rf.shape[0]) @ rf.reshape(rf.shape[0] * rf.shape[1], -1)
        return image.reshape(self.shape + rf.shape[2:])
