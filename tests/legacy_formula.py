"""What dasIT 1.0 / usdatarecycler compute in the beamforming step, written as closed formulas, with a switch for
each of its two numerical habits. Used to connect the golden frames to the new core link by link:

    golden frame == legacy_frame(paths float32, read_sample_0=True)      what the old code does, exactly
                    legacy_frame(paths float64, read_sample_0=True)      link A: path lengths in double precision
                    legacy_frame(paths float64, read_sample_0=False)     link B: a sample beyond the record is nothing
                 == new core with the legacy geometry and aperture
"""
import numpy as np


def legacy_geometry(n_elements=192, pitch=0.00023, fc=5208300.0, c=1540.0, adc_ratio=4, depth_wavelengths=177,
                    elevation_focus=0.028):
    """Element positions, pixel grid and aperture of the old classes for the example data."""
    fs = adc_ratio * fc
    x = np.linspace(-pitch * n_elements / 2, pitch * n_elements / 2, n_elements)       # spacing pitch * N / (N - 1)
    z = np.linspace(0, c / fc * depth_wavelengths, depth_wavelengths * adc_ratio, endpoint=False)
    f_number = 1 / (2 * ((pitch * n_elements / 2) / elevation_focus))
    active = np.ceil((z / (2 * f_number)) * n_elements / (pitch * n_elements)) // 2 * 2 + 1   # odd number of elements
    return dict(x=x, z=z, fs=fs, c=c, active=active, spacing=x[1] - x[0], n=n_elements)


def legacy_weights(g, window):
    """[depth, channel, pixel]: box of `active` elements, or np.hanning(active) (whose two end points are 0)."""
    offset = np.abs(np.arange(g["n"])[:, None] - np.arange(g["n"])[None, :])
    half = ((g["active"] - 1) // 2).astype(int)
    if window == "box":
        return (offset[None] <= half[:, None, None]).astype(float)
    w = np.zeros((len(g["z"]), g["n"], g["n"]))
    for k in np.unique(g["active"]).astype(int):
        w[g["active"] == k] = np.where(offset <= (k - 1) // 2, np.hanning(k)[np.minimum((k - 1) // 2 + offset, k - 1)], 0.0)
    return w


def legacy_delays(g, paths=np.float32):
    """[depth, channel, pixel] two-way delay in samples, rounded. The old code holds the path lengths in float32."""
    x, z = g["x"], g["z"]
    tx = np.broadcast_to(z[:, None, None], (len(z), g["n"], g["n"])).astype(paths)
    rx = np.sqrt(z[:, None, None] ** 2 + (x[None, :, None] - x[None, None, :]) ** 2).astype(paths)
    return np.rint(np.multiply((tx + rx) / g["c"], g["fs"])).astype(np.int64)


def legacy_frame(analytic, g, window, limit, paths=np.float32, read_sample_0=True, mask_on_index=False):
    """Beamformed frame [depth, pixel]. limit: delays above it are 'out of the record'.
    read_sample_0: the old code replaces such a delay by sample 0 and still adds that sample (True), or adds nothing.
    mask_on_index: dasIT 1.0 (main) applied the box by setting the delay to 0, so elements outside the aperture add
    sample 0 too."""
    d, w = legacy_delays(g, paths), legacy_weights(g, window)
    beyond = d > limit
    d = np.where(beyond, 0, d)
    channel = np.arange(g["n"])[None, :, None]
    if mask_on_index:
        return analytic[d * (w != 0), channel].sum(axis=1)
    if not read_sample_0:
        w = np.where(beyond, 0.0, w)
    return (analytic[d, channel] * w).sum(axis=1)


def core_settings(g, window):
    """The legacy geometry and aperture as inputs of dasIT.beamform.Beamformer (with interp='nearest', t0 = 0)."""
    half_elements = (g["active"] - 1) / 2
    half_aperture = (half_elements + (0.5 if window == "box" else 0.0)) * g["spacing"]     # Hann: weight 0 at the last element
    return dict(element_x=g["x"], grid_x=g["x"], grid_z=g["z"], fs=g["fs"], c=g["c"], window=window, half_aperture=half_aperture)
