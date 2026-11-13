"""The steps around the beamformer: filter, oversampling, log compression, loader, plot, and two properties of the
beamformer settings that the example relies on.
"""
import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from dasIT import signal as ds
from dasIT.beamform import Beamformer
from dasIT.data.loader import load_rf
from dasIT.geometry import linear_array, plane_wave
from dasIT.plot import show_image


def test_bandpass_rejects_dc_and_keeps_the_band():
    fs, n = 20.8332e6, 4096
    t = np.arange(n) / fs
    tone = lambda f: np.sin(2 * np.pi * f * t)[:, None]
    out = ds.bandpass(tone(5.2e6) + 3.0 + tone(0.3e6), (3.3125e6, 7.2875e6), fs)[500:-500, 0]
    assert abs(out.mean()) < 1e-3                                   # the offset is gone
    assert np.corrcoef(out, tone(5.2e6)[500:-500, 0])[0, 1] > 0.999  # the in-band tone is kept, without a time shift


def test_log_compress_range_and_reference():
    env = np.array([[1.0, 0.1], [0.0, 10.0]])
    db = ds.log_compress(env, 40)
    assert db.max() == 0 and db.min() == -40 and db[0, 0] == pytest.approx(-20)
    assert ds.log_compress(env, 40, reference=100.0)[1, 1] == pytest.approx(-20)


def test_log_compress_of_an_empty_frame_is_black():
    db = ds.log_compress(np.zeros((4, 3)), 55)                      # a frame that was not stored
    assert not np.isnan(db).any() and np.all(db == -55)


def test_oversample_keeps_the_original_samples():
    x = np.sin(2 * np.pi * 5e6 * np.arange(256) / 20e6 + 0.3)[:, None] * np.hanning(256)[:, None]
    assert np.allclose(ds.oversample(x, 4)[::4], x, atol=1e-9)


def test_loader_keeps_shots_and_frames_apart(tmp_path):
    path = tmp_path / "two_shots.h5"
    with h5py.File(path, "w") as f:
        for frame in range(3):
            for shot in range(2):
                f[f"frame{frame:04}/shot{shot:04}"] = np.full((16, 4), 10 * frame + shot, float)
    rf = load_rf(path)
    assert rf.shape == (16, 4, 2, 3)                                # samples, channels, shots, frames
    assert rf[0, 0, 1, 2] == 21 and rf[0, 0, 0, 1] == 10


def test_show_image_is_centred_and_leaves_its_inputs_alone():
    x, z = linear_array(8, 1e-3), np.arange(10) * 0.5e-3
    image = -np.random.default_rng(0).random((10, 8)) * 40
    x0, z0, i0 = x.copy(), z.copy(), image.copy()
    ax = show_image(image, x, z, depth_range=(1e-3, 3e-3))
    left, right, bottom, top = ax.images[0].get_extent()
    assert left == pytest.approx(-right) and (top, bottom) == pytest.approx((0.75, 3.25))   # mm, only the asked depths
    assert np.array_equal(x, x0) and np.array_equal(z, z0) and np.array_equal(image, i0)
    matplotlib.pyplot.close("all")


def test_aperture_uses_every_element_inside_it():
    elements = linear_array(192, 0.23e-3)
    z = np.arange(708) * 74e-6
    bf = Beamformer(elements, elements, z, 20.8332e6, 1540.0, tx=plane_wave(0), f_number=1.5, window="hann")
    row, column = 400, 96
    weights = np.array([bf.weights(e)[row, column] for e in range(elements.size)])
    inside = np.abs(elements - elements[column]) < z[row] / (2 * 1.5)
    assert np.array_equal(weights > 0, inside)


def test_interpolation_error_is_small():
    """On a synthetic record with 4 samples per period: nearest sample against 4x oversampling + linear."""
    fs, fc, c, n = 20e6, 5e6, 1540.0, 1200
    elements = linear_array(32, 0.3e-3)
    t = np.arange(n)[:, None] / fs
    arrival = (20e-3 + np.sqrt(20e-3 ** 2 + elements[None, :] ** 2)) / c          # a point at (0, 20 mm), plane wave at 0 degrees
    pulse = lambda time: np.exp(-0.5 * ((time - arrival) * fc) ** 2) * np.exp(2j * np.pi * fc * (time - arrival))
    grid_z = np.arange(18e-3, 22e-3, c / fs / 8)
    kw = dict(element_x=elements, grid_x=np.array([0.0]), grid_z=grid_z, c=c, tx=plane_wave(0), f_number=1.5, window="box")
    exact = np.abs(Beamformer(fs=64 * fs, interp="nearest", **kw)(pulse(np.arange(64 * n)[:, None] / (64 * fs)))).max()
    nearest = np.abs(Beamformer(fs=fs, interp="nearest", **kw)(pulse(t))).max()
    fine = np.abs(Beamformer(fs=4 * fs, interp="linear", **kw)(ds.oversample(pulse(t), 4))).max()
    assert abs(fine / exact - 1) < 0.02
    assert abs(nearest / exact - 1) > 5 * abs(fine / exact - 1)
