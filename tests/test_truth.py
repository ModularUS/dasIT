"""Truth tests: the beamformer on data with a known answer (tests/synthetic.py), independent of any stored result.

T1  a point scatterer appears at its position: on axis, off axis, shallow, deep
T2  the same for a plane wave steered to +10 and -10 degrees
T3  the lateral width of a point follows wavelength x f-number
T4  a wrong time zero moves the point by c * t0 / 2 in depth
T5  an echo that would arrive before the record starts contributes nothing (no sample is taken from the other end)
T6  every element inside a Hann aperture has a non-zero weight
T7  frames at once equal frame by frame, and both engines give the same image
T8  with normalisation the brightness of a point does not depend on depth or on the image edge
"""
import functools

import numpy as np
import pytest

from dasIT.beamform import Beamformer
from dasIT.geometry import linear_array, plane_wave
from synthetic import point_echoes, width_at_half_maximum

C, FC, FS = 1540.0, 5.0e6, 40.0e6
PITCH, N_EL, N_SAMPLES = 0.3e-3, 96, 2400
WAVELENGTH = C / FC
ELEMENTS = linear_array(N_EL, PITCH)
DX, DZ = PITCH / 2, C / FS                                             # pixel size of the test images
GRID_X = np.arange(-12e-3, 12e-3 + 1e-9, DX)
GRID_Z = np.arange(4e-3, 40e-3, DZ)
POINTS = [(0.0, 10e-3), (6e-3, 15e-3), (-9e-3, 30e-3), (2.4e-3, 6e-3), (-4.5e-3, 36e-3)]


@functools.lru_cache(maxsize=None)                                       # one operator per setting, shared by the tests
def _beamformer(angle_deg, settings):
    return Beamformer(ELEMENTS, GRID_X, GRID_Z, FS, C, tx=plane_wave(angle_deg), **dict(settings))


def beamformer(angle_deg=0.0, **kw):
    kw = dict(dict(f_number=1.5, window='hann', interp='linear', engine='matrix'), **kw)
    return _beamformer(angle_deg, tuple(sorted(kw.items())))


def echoes(points, angle_deg=0.0, **kw):
    return point_echoes(points, ELEMENTS, FS, N_SAMPLES, C, plane_wave(angle_deg), FC, **kw)


def peak(image):
    iz, ix = np.unravel_index(np.argmax(np.abs(image)), image.shape)
    return GRID_X[ix], GRID_Z[iz]


@pytest.mark.parametrize("angle", [0.0, 10.0, -10.0])                    # T1 (0 degrees) and T2 (steered)
@pytest.mark.parametrize("point", POINTS)
def test_point_appears_at_its_position(point, angle):
    x, z = peak(beamformer(angle)(echoes([point], angle)))
    assert abs(x - point[0]) <= DX + 1e-9                                  # one lateral pixel
    assert abs(z - point[1]) <= DZ + 1e-9                                  # one axial pixel


@pytest.mark.parametrize("f_number", [1.0, 1.5, 2.5])                     # T3
def test_lateral_width_follows_wavelength_times_f_number(f_number):
    point = (0.0, 25e-3)
    fine_x = np.arange(-3e-3, 3e-3, PITCH / 16)                            # one image row through the point, finely sampled
    bf = Beamformer(ELEMENTS, fine_x, np.array([point[1]]), FS, C, tx=plane_wave(0), f_number=f_number, window='box')
    width = width_at_half_maximum(bf(echoes([point]))[0], PITCH / 16)
    assert width == pytest.approx(1.2 * WAVELENGTH * f_number, rel=0.30)   # box aperture: 1.2 x wavelength x F


def test_time_zero_moves_the_point_in_depth():                             # T4
    point, t0 = (0.0, 20e-3), 2.0e-6                                       # the record starts 2 us after the transmit
    data = echoes([point], t0=t0)
    assert peak(beamformer(t0=t0)(data))[1] == pytest.approx(point[1], abs=DZ + 1e-9)
    shifted = peak(beamformer(t0=0.0)(data))[1]                            # the beamformer believes the record starts at 0
    assert point[1] - shifted == pytest.approx(C * t0 / 2, abs=2 * DZ)


@pytest.mark.parametrize("engine", ["matrix", "loop"])                     # T5
@pytest.mark.parametrize("interp", ["nearest", "linear"])
def test_samples_before_the_record_contribute_nothing(engine, interp):
    data = np.zeros((N_SAMPLES, N_EL))
    data[-40:] = 1.0                                                       # only the end of the record holds signal
    t0 = 30e-6                                                             # record starts late: shallow echoes came before it
    bf = beamformer(t0=t0, engine=engine, interp=interp, window='box')
    image = bf(data)
    before = np.max([bf.delays(e) for e in (0, N_EL // 2, N_EL - 1)], axis=0) < -1    # pixels whose echo precedes the record
    assert before.any() and np.all(image[before] == 0)


def test_every_element_inside_a_hann_aperture_counts():                    # T6
    bf = beamformer(window='hann', f_number=1.5)
    weights = np.array([bf.weights(e) for e in range(N_EL)])               # [element, depth, lateral]
    column = int(np.argmin(np.abs(GRID_X)))
    for row in (50, 450, 900):
        half = max(GRID_Z[row] / (2 * 1.5), PITCH / 2)
        inside = np.abs(ELEMENTS - GRID_X[column]) < half
        assert np.array_equal(weights[:, row, column] > 0, inside) and inside.sum() >= 1


def test_frames_and_engines_agree():                                       # T7
    frames = np.stack([echoes([POINTS[k]]) for k in range(3)], axis=2)     # [samples, channels, 3 frames]
    matrix, loop = beamformer(engine='matrix'), beamformer(engine='loop')
    together = matrix(frames)
    assert together.shape == (GRID_Z.size, GRID_X.size, 3)
    scale = np.abs(together).max()
    for k in range(3):
        assert np.abs(together[:, :, k] - matrix(frames[:, :, k])).max() <= 1e-12 * scale
    assert np.abs(together - loop(frames)).max() <= 1e-12 * scale
    assert np.abs(matrix(frames.real) - together.real).max() <= 1e-12 * scale     # real RF in, real part out


@pytest.mark.parametrize("engine", ["matrix", "loop"])                     # T8
def test_normalised_brightness_is_independent_of_depth_and_edge(engine):
    points = [(0.0, 8e-3), (0.0, 32e-3), (-11.4e-3, 20e-3)]                # shallow, deep, at the image edge
    bf = beamformer(normalise=True, engine=engine, window='box')
    peaks = [np.abs(bf(echoes([p]))).max() for p in points]
    assert max(peaks) / min(peaks) < 1.15
    raw = [np.abs(beamformer(normalise=False, engine=engine, window='box')(echoes([p]))).max() for p in points[:2]]
    assert raw[1] / raw[0] > 2                                             # without it the deep point is much brighter


def test_unknown_settings_name_the_choices():
    with pytest.raises(ValueError, match="window"):
        beamformer(window='blackman-harris')
    with pytest.raises(ValueError, match="channels"):
        beamformer()(np.zeros((N_SAMPLES, N_EL + 1)))
