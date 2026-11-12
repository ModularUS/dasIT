"""The corrections of stage 3: every golden generation reproduced, each change of the size and kind that was
predicted for it, and one test per corrected defect of the helpers.

The generations (tests/generations.py, tests/golden/LINEAGE.md) are recomputed once per test run and compared with
tests/golden/current/lineage.json: by SHA-256 with the library versions they were made with, by their measurements
otherwise.
"""
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import scipy
from scipy import signal as sps

import generations as G
from dasIT import signal as ds
from dasIT.beamform import Beamformer
from dasIT.data.loader import load_rf
from dasIT.geometry import linear_array, plane_wave
from dasIT.plot import show_image

pytestmark = pytest.mark.skipif(not (G.EXAMPLE / "CIRS_phantom.h5").exists(), reason="example data not present")


@pytest.fixture(scope="module")
def stored():
    return json.loads((G.CURRENT / "lineage.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frames():
    """{name: (frame, grid_x, grid_z)} of v1 .. v10, computed now."""
    out = {"v1": G.legacy_v1()}
    out.update({name: G.generation(name) for name in G.CHANGES})
    return out


def entry(stored, name):
    return next(e for e in stored["generations"] if e["name"] == name)


# ---- the lineage as a whole
def test_every_generation_is_reproduced(frames, stored):
    same_libraries = stored["versions"] == dict(numpy=np.__version__, scipy=scipy.__version__)
    for name in G.CHANGES:
        frame, x, z = frames[name]
        e = entry(stored, name)
        assert list(frame.shape) == e["shape"], name
        if same_libraries:
            assert G.sha(frame) == e["sha256"], f"{name}: result changed"
        measured = G.measure(np.abs(frame), x, z)
        for key in ("lateral_scale_percent", "depth_scale_percent", "depth_offset_mm", "cylinder_contrast_db", "speckle_snr"):
            assert measured[key] == pytest.approx(e["phantom"][key], abs=0.02), f"{name} {key}"


def test_newest_generation_is_the_stored_golden(frames):
    stored_frame = np.load(G.CURRENT / "golden.npz")["beamformed"]
    newest = frames[list(G.CHANGES)[-1]][0]
    assert np.abs(newest - stored_frame).max() <= 2e-6 * np.abs(stored_frame).max()


# ---- each change, as predicted
def change(frames, name):
    names = ["v1"] + list(G.CHANGES)
    return G.compare(frames[name][0], frames[names[names.index(name) - 1]][0])


def test_v2_and_v3_barely_move_the_image(frames):
    assert change(frames, "v2")["mean_over_mean"] < 1e-3            # links A and B of tests/test_core_golden.py
    assert change(frames, "v3")["mean_over_mean"] < 1e-3            # the longer record only changes the analytic signal slightly ...
    full, crop = Beamformer(tx=plane_wave(0), **G.settings("v3")[1]), None
    bottom = np.asarray((full.matrix(4096) != 0).sum(axis=1)).reshape(full.shape)[-20:].sum()
    cropped = np.asarray((full.matrix(G.CROP) != 0).sum(axis=1)).reshape(full.shape)[-20:].sum()
    assert bottom > cropped                                         # ... and gives the deepest rows their whole aperture


def test_v4_corrects_the_lateral_scale(frames, stored):
    before = G.measure(np.abs(frames["v3"][0]), *frames["v3"][1:])["lateral_scale_percent"]
    after = G.measure(np.abs(frames["v4"][0]), *frames["v4"][1:])["lateral_scale_percent"]
    assert before == pytest.approx(100 * (192 / 191 - 1), abs=0.05)  # the stretch of the old element positions
    assert abs(after) < 0.1                                          # the 6 mm wire spacing of the phantom is met


def test_v5_uses_every_element_of_the_aperture():
    bf = Beamformer(tx=plane_wave(0), **G.settings("v5")[1])
    row, column = 400, 96
    weights = np.array([bf.weights(e)[row, column] for e in range(G.N_ELEMENTS)])
    inside = np.abs(bf.element_x - bf.grid_x[column]) < bf.grid_z[row] / (2 * G.F_NUMBER)
    assert np.array_equal(weights > 0, inside)


def test_v6_interpolation_error_is_small():
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


def test_v7_removes_the_growth_of_brightness_with_depth(frames):
    ratio = lambda name: G.measure(np.abs(frames[name][0]), *frames[name][1:])["brightness_40_over_10_mm_db"]
    assert ratio("v7") < ratio("v6") - 6                           # the aperture is more than twice as wide at 47 mm as at 12 mm


def test_v10_is_v9_on_a_finer_lateral_grid(frames):
    coarse, fine = frames["v9"][0], frames["v10"][0]
    assert fine.shape == (coarse.shape[0], G.FINE * (coarse.shape[1] - 1) + 1)
    assert np.abs(fine[:, ::G.FINE] - coarse).max() <= 1e-9 * np.abs(coarse).max()   # the old pixels are unchanged
    assert np.all(np.abs(fine) >= 0)                               # an envelope from beamforming cannot be negative


# ---- the corrected helpers
def test_bandpass_rejects_dc_and_keeps_the_band():
    fs, n = 20.8332e6, 4096
    t = np.arange(n) / fs
    tone = lambda f: np.sin(2 * np.pi * f * t)[:, None]
    out = ds.bandpass(tone(5.2e6) + 3.0 + tone(0.3e6), G.BAND_HZ, fs)[500:-500, 0]
    assert abs(out.mean()) < 1e-3                                   # the offset is gone
    assert np.corrcoef(out, tone(5.2e6)[500:-500, 0])[0, 1] > 0.999  # the in-band tone is kept, without a time shift
    old = np.load(G.LEGACY / "golden.npz")["filter_coefficients"]
    assert abs(old.sum()) > 0.3                                     # the v1.0 filter let 36 % of an offset through


def test_log_compress_range_and_reference():
    env = np.array([[1.0, 0.1], [0.0, 10.0]])
    db = ds.log_compress(env, 40)
    assert db.max() == 0 and db.min() == -40 and db[0, 0] == pytest.approx(-20)
    assert ds.log_compress(env, 40, reference=100.0)[1, 1] == pytest.approx(-20)


def test_tgc_gain_is_in_decibel_and_can_be_undone():
    gain = ds.tgc_gain([0, 1023], 5)
    assert gain[0] == pytest.approx(1.0) and gain[-1] == pytest.approx(100.0)       # 40 dB
    rf = np.ones((5, 3, 2))
    assert np.allclose(ds.apply_gain(ds.apply_gain(rf, gain), gain, undo=True), rf)


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
