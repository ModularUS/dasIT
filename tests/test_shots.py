"""Several transmits for one image, and scanner-beamformed lines.

S1  a point seen by three steered plane waves lands in one place, on the truth
S2  one operator for all shots equals beamforming every shot alone and adding the images, for both engines
S3  compounding three angles lowers the side lobes of a point against one angle
S4  the data layouts with shots and frames, and the errors for a wrong layout
S5  line data: axes in metres that match the lines, through envelope, log compression and the plot
"""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from dasIT import signal as ds
from dasIT.beamform import Beamformer
from dasIT.geometry import depth_axis, line_positions, linear_array, plane_wave, plane_waves
from dasIT.plot import show_image
from synthetic import point_echoes

C, FC, FS = 1540.0, 5.0e6, 40.0e6
PITCH, N_EL, N_SAMPLES = 0.3e-3, 96, 2400
ELEMENTS = linear_array(N_EL, PITCH)
GRID_X = np.arange(-9e-3, 9e-3 + 1e-9, PITCH / 2)
GRID_Z = np.arange(8e-3, 30e-3, C / FS)
ANGLES = [-10.0, 0.0, 10.0]
KW = dict(f_number=1.5, window='hann', interp='linear')


def shots(points, waves):
    """[samples, channels, shots]: the same scatterers insonified by every wave."""
    return np.stack([point_echoes(points, ELEMENTS, FS, N_SAMPLES, C, wave, FC) for wave in waves], axis=2)


@pytest.mark.parametrize("timing", ["centre", "first element"])
@pytest.mark.parametrize("point", [(0.0, 12e-3), (4.5e-3, 20e-3), (-6e-3, 26e-3)])
def test_s1_compounded_point_is_at_its_position(point, timing):
    waves = plane_waves(ANGLES, ELEMENTS if timing == "first element" else None)
    image = Beamformer(ELEMENTS, GRID_X, GRID_Z, FS, C, tx=waves, **KW)(shots([point], waves))
    iz, ix = np.unravel_index(np.argmax(np.abs(image)), image.shape)
    assert abs(GRID_X[ix] - point[0]) <= PITCH / 2 + 1e-9 and abs(GRID_Z[iz] - point[1]) <= C / FS + 1e-9


@pytest.mark.parametrize("engine", ["matrix", "loop"])
def test_s2_one_operator_equals_the_sum_of_the_single_shots(engine):
    waves = plane_waves(ANGLES, ELEMENTS)
    data = shots([(0.0, 12e-3), (4.5e-3, 20e-3)], waves)
    together = Beamformer(ELEMENTS, GRID_X, GRID_Z, FS, C, tx=waves, engine=engine, **KW)(data)
    alone = sum(Beamformer(ELEMENTS, GRID_X, GRID_Z, FS, C, tx=wave, engine=engine, **KW)(data[:, :, k]) for k, wave in enumerate(waves))
    assert np.abs(together - alone).max() <= 1e-12 * np.abs(alone).max()


def test_s3_compounding_lowers_the_side_lobes():
    point = (0.0, 20e-3)
    row = np.array([point[1]])
    x = np.arange(-8e-3, 8e-3, PITCH / 8)

    def side_lobe_db(angles):
        waves = plane_waves(angles)
        profile = np.abs(Beamformer(ELEMENTS, x, row, FS, C, tx=waves, f_number=1.5, window='box')(shots([point], waves)))[0]
        outside = np.abs(x - point[0]) > 1.5e-3                             # beyond the main lobe
        return 20 * np.log10(profile[outside].max() / profile.max())

    single, compounded = side_lobe_db([0.0]), side_lobe_db([-12.0, -6.0, 0.0, 6.0, 12.0])
    assert compounded < single - 3                                          # at least 3 dB lower


def test_s4_layouts():
    waves = plane_waves(ANGLES)
    data = shots([(0.0, 12e-3)], waves)
    bf = Beamformer(ELEMENTS, GRID_X, GRID_Z, FS, C, tx=waves, **KW)
    frames = np.stack([data, 2 * data], axis=3)                             # [samples, channels, shots, 2 frames]
    out = bf(frames)
    assert out.shape == (GRID_Z.size, GRID_X.size, 2) and np.allclose(out[:, :, 1], 2 * bf(data))
    one = Beamformer(ELEMENTS, GRID_X, GRID_Z, FS, C, tx=plane_wave(0), **KW)
    single = data[:, :, 1]                                                  # the shot at 0 degrees
    assert np.allclose(one(single[:, :, None, None])[:, :, 0], one(single))  # a shot axis of length 1 is accepted
    with pytest.raises(ValueError, match="shots"):
        bf(data[:, :, :2])                                                  # three shots expected, two given
    with pytest.raises(ValueError, match="shots"):
        bf(data[:, :, 0])


def test_s5_line_data_axes_and_display():
    n_lines, n_samples, fs = 64, 780, 20e6
    x, z = line_positions(ELEMENTS, n_lines), depth_axis(n_samples, fs, C)
    assert x[0] == ELEMENTS[0] and x[-1] == ELEMENTS[-1] and len(x) == n_lines
    assert z[0] == 0 and z[-1] == pytest.approx((n_samples - 1) * C / (2 * fs))
    assert depth_axis(4, fs, C, t0=1e-6)[0] == pytest.approx(C * 1e-6 / 2)
    t = np.arange(n_samples)[:, None] / fs
    lines = np.exp(-0.5 * ((t - 2 * 15e-3 / C) * 3e6) ** 2) * np.cos(2 * np.pi * 5e6 * t) * np.ones((1, n_lines))   # a layer at 15 mm
    db = ds.log_compress(ds.envelope(ds.analytic(lines)), 40)
    assert z[np.argmax(db[:, 10])] == pytest.approx(15e-3, abs=0.1e-3)
    ax = show_image(db, x, z)
    left, right, bottom, top = ax.images[0].get_extent()
    assert left == pytest.approx(-right) and top < 0.1 and bottom == pytest.approx(1e3 * z[-1], abs=0.1)
    matplotlib.pyplot.close("all")
