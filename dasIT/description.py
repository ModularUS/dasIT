'''
Licensed under the Apache License, Version 2.0, see the LICENSE file of the dasIT repository.

A beamforming description: everything the beamformer has to be told about a recording, as one plain dictionary that
can be stored beside the data (JSON, YAML, attributes of an h5 file) and filled by any acquisition system.

    description = {
        "element_x":    [...],              x of the receive elements [m], one per data channel
        "fs":           20e6,               sampling rate [Hz]
        "c":            1540.0,             speed of sound [m/s]
        "transmits":    [{"angle_deg": 0.0, "x0": 0.0}],   one plane wave per shot; x0: where its clock starts [m]
        "t0":           0.0,                time of sample 0 on that clock [s]                      (optional)
        "band_hz":      [3e6, 7e6],         band-pass before beamforming, None for none             (optional)
        "f_number":     1.5,                receive aperture                                        (optional)
        "window":       "box",              "box" or "hann"                                         (optional)
        "interp":       "linear",           "linear" or "nearest"                                   (optional)
        "first_frame":  0,                  frames before this one are not to be used               (optional)
        "skip_samples": 0,                  samples at the start of a channel to set to zero:
                                            one number, or one per channel                          (optional)
    }

    rf = prepare(rf, description)                     band-pass, start of the channels blanked, shots on their own axis
    bf = beamformer(description, n_samples=rf.shape[0])
    image = bf(rf)

Every key can be overridden when the beamformer is made (beamformer(description, f_number=1.0)), which is how an
interactive tool changes one setting and keeps the rest.
'''

import json

import numpy as np

from dasIT.beamform import ENGINES, INTERPOLATIONS, WINDOWS, Beamformer
from dasIT.geometry import depth_axis, plane_wave
from dasIT.signal import bandpass

REQUIRED = ("element_x", "fs", "c", "transmits")
DEFAULTS = dict(t0=0.0, band_hz=None, f_number=1.5, window="box", interp="linear", first_frame=0, skip_samples=0)


def complete(description, **overrides):
    """The description with its defaults filled in and the overrides applied, checked. Raises ValueError that names
    the key that is missing or wrong."""
    unknown = set(description) | set(overrides)
    unknown -= set(REQUIRED) | set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown key(s) in the beamforming description: {sorted(unknown)}")
    d = dict(DEFAULTS, **description)
    d.update(overrides)
    for key in REQUIRED:
        if key not in d:
            raise ValueError(f"the beamforming description lacks '{key}'")
    d["element_x"] = [float(x) for x in np.ravel(d["element_x"])]
    n = len(d["element_x"])
    if n == 0:
        raise ValueError("'element_x' is empty")
    for key in ("fs", "c", "f_number"):
        if not float(d[key]) > 0:
            raise ValueError(f"'{key}' must be positive, got {d[key]}")
    if not d["transmits"] or not all(isinstance(t, dict) and "angle_deg" in t for t in d["transmits"]):
        raise ValueError("'transmits' must be a list with one {'angle_deg': ..., 'x0': ...} per shot")
    if d["band_hz"] is not None:
        low, high = (float(v) for v in d["band_hz"])
        if not 0 < low < high < float(d["fs"]) / 2:
            raise ValueError(f"'band_hz' must lie between 0 and half the sampling rate, got {d['band_hz']}")
        d["band_hz"] = [low, high]
    if d["window"] not in WINDOWS or d["interp"] not in INTERPOLATIONS:
        raise ValueError(f"'window' must be one of {sorted(WINDOWS)} and 'interp' one of {list(INTERPOLATIONS)}")
    skip = np.ravel(d["skip_samples"]).astype(int)
    if skip.size not in (1, n) or skip.min() < 0:
        raise ValueError(f"'skip_samples' must be one number or one per channel ({n}), not negative")
    d["skip_samples"] = int(skip[0]) if skip.size == 1 else skip.tolist()
    if int(d["first_frame"]) < 0:
        raise ValueError("'first_frame' must not be negative")
    return d


def prepare(rf, description, silent=None, **overrides):
    """Channel data ready for the beamformer: the silent reference taken off, the start of the channels set to zero,
    the band-pass applied. Returns a new float array.

    rf: [samples, channels(, frames)] for one transmit, [samples, channels, shots(, frames)] for several. Several
    transmits may also come as [samples, shots * channels(, frames)], the columns of one shot after those of the
    one before, as a scanner writes them; they are then put on their own axis.
    silent: what the system records without transmitting (electrical pickup), same layout as one frame, or None."""
    d = complete(description, **overrides)
    n, shots = len(d["element_x"]), len(d["transmits"])

    def channels_first(x):
        x = np.array(x, dtype=np.float64)
        if shots > 1 and x.shape[1] == n * shots:
            x = np.moveaxis(x.reshape((x.shape[0], shots, n) + x.shape[2:]), 1, 2)
        if x.shape[1] != n or (shots > 1 and (x.ndim < 3 or x.shape[2] != shots)):
            raise ValueError(f"rf has the shape {x.shape}; the description has {n} channels and {shots} shot(s)")
        return x

    rf = channels_first(rf)
    if silent is not None:
        silent = channels_first(silent)
        rf -= silent.reshape(silent.shape + (1,) * (rf.ndim - silent.ndim))
    skip = np.broadcast_to(d["skip_samples"], (n,))
    if skip.any():
        rf[np.arange(rf.shape[0])[:, None] < skip[None, :]] = 0
    return bandpass(rf, tuple(d["band_hz"]), float(d["fs"])) if d["band_hz"] else rf


def beamformer(description, n_samples=None, grid_x=None, grid_z=None, engine="matrix", **overrides):
    """A Beamformer for the recording. Without a grid: one lateral pixel per element and one axial pixel per sample
    (n_samples needed), the depth of each sample following from fs, c and t0."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {list(ENGINES)}")
    d = complete(description, **overrides)
    element_x = np.asarray(d["element_x"])
    if grid_z is None:
        if n_samples is None:
            raise ValueError("give grid_z, or n_samples for one axial pixel per sample")
        grid_z = np.maximum(depth_axis(n_samples, float(d["fs"]), float(d["c"]), float(d["t0"])), 0.0)
    tx = [plane_wave(float(t["angle_deg"]), float(t.get("x0", 0.0))) for t in d["transmits"]]
    return Beamformer(element_x, element_x if grid_x is None else grid_x, grid_z, float(d["fs"]), float(d["c"]),
                      tx=tx if len(tx) > 1 else tx[0], t0=float(d["t0"]), f_number=float(d["f_number"]),
                      window=d["window"], interp=d["interp"], engine=engine)


def dumps(description):
    """The checked description as JSON text, for a file or an attribute."""
    return json.dumps(complete(description), indent=1)


def loads(text):
    return complete(json.loads(text))
