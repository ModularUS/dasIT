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
    env = np.asarray(env, dtype=float)
    reference = np.max(env) if reference is None else reference
    if not reference > 0:                                   # an empty frame: everything at the lower limit
        return np.full(env.shape, -float(dynamic_range_db))
    with np.errstate(divide='ignore'):
        db = 20 * np.log10(env / reference)
    return np.clip(db, -dynamic_range_db, 0.0)
