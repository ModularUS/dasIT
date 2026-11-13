"""The example frame (CIRS 054GS phantom, GE 9L-D, Verasonics Vantage, one plane wave at 0 degrees) through the chain
that beamform_image.ipynb teaches. The values come from the acquisition settings of the recording.

    band-pass 3.3 - 7.3 MHz  ->  analytic signal  ->  4x oversampling  ->  Beamformer (f/1.5, Hann, linear, normalised)
"""
from pathlib import Path

import numpy as np

from dasIT.beamform import Beamformer
from dasIT.data.loader import load_rf
from dasIT.geometry import linear_array, plane_wave
from dasIT.signal import analytic, bandpass, oversample

DATA = Path(__file__).resolve().parents[1] / "example_data" / "CIRSphantom_GE9LD_VVantage" / "CIRS_phantom.h5"
FC, C = 5208300.0, 1540.0                              # transmit centre frequency [Hz], speed of sound [m/s]
FS = 4 * FC                                            # four samples per period
BAND_HZ = (3312500.0, 7287500.0)                       # band edges of the transducer
N_ELEMENTS, PITCH = 192, 0.23e-3
T0 = 2 * (5.0 - 3.3820129870129874) / FC               # s: recording starts at 5 wavelengths depth, the lens delays by 3.38
OVERSAMPLE, F_NUMBER, FINE = 4, 1.5, 3                 # FINE: lateral pixels per element


def grid():
    """(element_x, grid_x, grid_z) [m]: 177 wavelengths deep at four pixels per wavelength, three columns per element."""
    element_x = linear_array(N_ELEMENTS, PITCH)
    grid_x = np.linspace(element_x[0], element_x[-1], FINE * (N_ELEMENTS - 1) + 1)
    return element_x, grid_x, np.arange(177 * 4) * C / FC / 4


def example_frame(frame=0):
    """(beamformed frame [depth, lateral] complex, grid_x, grid_z)."""
    element_x, grid_x, grid_z = grid()
    signal = oversample(analytic(bandpass(load_rf(DATA)[:, :, 0, frame], BAND_HZ, FS)), OVERSAMPLE)
    beamformer = Beamformer(element_x, grid_x, grid_z, OVERSAMPLE * FS, C, tx=plane_wave(0), t0=T0, f_number=F_NUMBER,
                            window="hann", interp="linear", normalise=True)
    return beamformer(signal), grid_x, grid_z
