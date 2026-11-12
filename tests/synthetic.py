"""Channel data of point scatterers, computed from geometry alone. No dasIT code is used here, so an image
beamformed from these data can be checked against the positions that went in.

Every channel receives, from every point, one short pulse centred at

    arrival time = transmit delay to the point + distance(point, element) / c

The pulse is evaluated at the sample instants, so the arrival times are exact, not rounded to a sample.
"""
import numpy as np


def point_echoes(points, element_x, fs, n_samples, c, tx, fc, t0=0.0, amplitudes=None, cycles=1.0):
    """Analytic channel data [samples, channels] (complex; its real part is the RF) for scatterers at points [(x, z)].
    tx(x, z, c) is the transmit delay, fc the pulse centre frequency [Hz], t0 the time of sample 0,
    cycles the standard deviation of the Gaussian envelope in periods of fc."""
    t = t0 + np.arange(n_samples)[:, None] / fs                           # [samples, 1]
    element_x = np.asarray(element_x, float)[None, :]                     # [1, channels]
    sigma = cycles / fc
    data = np.zeros((n_samples, element_x.shape[1]), complex)
    for k, (x, z) in enumerate(points):
        arrival = tx(x, z, c) + np.sqrt(z ** 2 + (x - element_x) ** 2) / c
        amplitude = 1.0 if amplitudes is None else amplitudes[k]
        data += amplitude * np.exp(-0.5 * ((t - arrival) / sigma) ** 2) * np.exp(2j * np.pi * fc * (t - arrival))
    return data


def width_at_half_maximum(profile, spacing):
    """Full width at half maximum of the peak of a 1-D profile, in the unit of spacing (linear between samples)."""
    profile = np.abs(profile)
    peak = int(np.argmax(profile))
    half = profile[peak] / 2
    left = peak
    while left > 0 and profile[left] > half:
        left -= 1
    right = peak
    while right < len(profile) - 1 and profile[right] > half:
        right += 1
    lo = left + (half - profile[left]) / (profile[left + 1] - profile[left]) if profile[left] < half else left
    hi = right - (half - profile[right]) / (profile[right - 1] - profile[right]) if profile[right] < half else right
    return (hi - lo) * spacing
