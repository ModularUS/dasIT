'''
Licensed under the Apache License, Version 2.0, see the LICENSE file of the dasIT repository.

Signal steps around the beamformer, as plain functions on arrays [samples, channels(, frames)]. Time runs along
axis 0. Units are SI; gains and image values are in dB where the name says so.

    rf = bandpass(rf, (3.3e6, 7.3e6), fs)      before beamforming
    a  = analytic(rf)                          complex signal, its magnitude is the envelope
    image = Beamformer(...)(a)                 dasIT.beamform
    db = log_compress(envelope(image), 50)     for display
'''

import numpy as np
from scipy import signal as sps


def bandpass(rf, band, fs, order=4):
    """Zero-phase Butterworth band-pass along the time axis. band = (low, high) in Hz.
    Removes the DC offset and the noise outside the transducer band; no time shift."""
    sos = sps.butter(order, band, btype='bandpass', fs=fs, output='sos')
    return sps.sosfiltfilt(sos, rf, axis=0)


def analytic(rf):
    """Analytic signal along the time axis (Hilbert transform). Real part: the input. Magnitude: the envelope."""
    return sps.hilbert(rf, axis=0)


def oversample(rf, factor):
    """Band-limited resampling of the time axis to factor times the sampling rate (the new rate is factor * fs).
    Makes linear interpolation between samples accurate when the record has few samples per period."""
    return sps.resample(rf, rf.shape[0] * int(factor), axis=0)


def envelope(x):
    """Magnitude of an analytic (complex) signal or image."""
    return np.abs(x)


def log_compress(env, dynamic_range_db=50.0, reference=None):
    """Envelope -> dB below `reference` (default: the maximum of env), limited to [-dynamic_range_db, 0].
    Pass one reference for a series of frames to keep their brightness comparable."""
    reference = np.max(env) if reference is None else reference
    with np.errstate(divide='ignore'):
        db = 20 * np.log10(np.asarray(env, dtype=float) / reference)
    return np.clip(db, -dynamic_range_db, 0.0)


def tgc_gain(control_points, n_samples, max_db=40.0, max_value=1023.0):
    """Gain curve [n_samples] as a linear factor, from control points spread evenly over the record.
    The control values run from 0 to max_value and mean 0 to max_db decibel (Verasonics: 0..1023 for 0..40 dB)."""
    control_points = np.ravel(np.asarray(control_points, dtype=float))
    position = np.linspace(0, n_samples - 1, control_points.size)
    return 10 ** (np.interp(np.arange(n_samples), position, control_points) * max_db / max_value / 20)


def apply_gain(rf, gain, undo=False):
    """Multiply every channel by the gain curve [samples], or divide by it (undo=True) to remove a gain the
    scanner applied."""
    gain = np.asarray(gain, dtype=float).reshape((-1,) + (1,) * (np.ndim(rf) - 1))
    return rf / gain if undo else rf * gain
