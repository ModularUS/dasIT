"""The chain against the golden model (tests/golden/README.md).

    pytest                      from the repository root

G1  Hann window: every stage up to the interpolated envelope equals golden/usdatarecycler. With the library versions
    the golden model was made with the SHA-256 of each stage is compared, otherwise the values with a tolerance.
G2  The log image may differ from golden/usdatarecycler only where the interpolated envelope is negative: there the
    golden code took the logarithm of a negative number, envelope() now returns the magnitude.
G3  Box window: the beamformed frame equals the closed formula of a box-weighted sum.
G4  The box window differs from golden/main only by what main added for the elements outside the aperture (it read
    sample 0 of their channel instead of nothing).
G5  The filter gives the same result on [samples, channels] and on [samples, channels, frames].
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import scipy

from chain import EXAMPLE, run_chain
from dasIT.features.signal import RFfilter

GOLDEN = Path(__file__).parent / "golden"
pytestmark = pytest.mark.skipif(not (EXAMPLE / "CIRS_phantom.h5").exists(), reason="example data not present")
EXACT_STAGES = ("rf", "tgc", "filtered", "analytic", "apodization", "delays", "beamformed", "envelope", "interpolated")


def golden(branch):
    return np.load(GOLDEN / branch / "golden.npz"), json.loads((GOLDEN / branch / "golden.json").read_text(encoding="utf-8"))


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def same_libraries(manifest):
    return manifest["versions"]["numpy"] == np.__version__ and manifest["versions"]["scipy"] == scipy.__version__


def close(a, b, rel=2e-6):
    """Equal to the precision of the stored golden arrays (single precision), relative to the largest value."""
    return np.abs(np.asarray(a) - np.asarray(b)).max() <= rel * np.abs(b).max()


@pytest.fixture(scope="module")
def hann():
    return run_chain(apo="hanning")


@pytest.fixture(scope="module")
def box():
    return run_chain(apo="rec", max_depth_samples=1361)        # 1361: entries above sample 1360 are dropped, as on main


def test_g1_stages_equal_usdatarecycler_golden(hann):
    arrays, manifest = golden("usdatarecycler")
    for name in EXACT_STAGES:
        stage = manifest["stages"][name]
        assert list(np.shape(hann[name])) == stage["shape"], name
        if same_libraries(manifest):
            assert sha(hann[name]) == stage["sha256"], f"{name}: result changed"
        else:
            assert abs(np.abs(hann[name]).mean() - stage["mean"]) <= 1e-9 * stage["mean"], name
        if name in arrays:
            assert close(hann[name], arrays[name]), name


def test_g2_log_image_differs_only_at_negative_envelope(hann):
    arrays, _ = golden("usdatarecycler")
    negative = hann["interpolated"] < 0
    assert negative.sum() == (arrays["interpolated"] < 0).sum() == 225      # the cubic interpolation undershoots there
    assert np.array_equal(hann["log_image"][~negative], arrays["log_image"][~negative])
    assert np.all(arrays["log_image"][negative] == 0)                       # golden: logarithm of a negative number -> black


def box_formula(analytic, g, limit):
    """Sum over the receive channels of the analytic signal at the rounded two-way delay, inside a box aperture of
    z / (2 F) elements. Path lengths in single precision and samples above `limit` replaced by sample 0, as dasIT does."""
    x, z, n = g["x"], g["z"], g["n_elements"]
    tx = np.broadcast_to(z[:, None, None], (len(z), n, n)).astype(np.float32)
    rx = np.sqrt(z[:, None, None] ** 2 + (x[None, :, None] - x[None, None, :]) ** 2).astype(np.float32)
    d = np.rint(np.multiply((tx + rx) / g["c"], g["fs"]))
    d[d > limit] = 0
    f_number = 1 / (2 * ((g["pitch"] * n / 2) / g["elevation_focus"]))
    active = np.ceil((z / (2 * f_number)) * n / (g["pitch"] * n)) // 2 * 2 + 1
    offset = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    inside = offset[None] <= ((active - 1) / 2)[:, None, None]
    return (analytic[d.astype(np.int64), np.arange(n)[None, :, None]] * inside).sum(axis=1)


def test_g3_box_window_equals_the_formula(box):
    expected = box_formula(box["analytic"], box["_geometry"], limit=1360)
    assert np.abs(box["beamformed"] - expected).max() <= 1e-12 * np.abs(expected).max()
    weights = np.asarray(box["apodization"])
    assert set(np.unique(weights)) == {0.0, 1.0}


def test_g4_box_window_against_main_golden(box):
    arrays, _ = golden("main")
    assert close(box["analytic"], arrays["analytic"])            # identical up to the beamformer
    difference = np.abs(box["beamformed"] - arrays["beamformed"])
    assert 3e-4 < difference.max() / np.abs(arrays["beamformed"]).max() < 5e-4
    assert 0.005 < difference.mean() / np.abs(box["beamformed"]).mean() < 0.02


def test_g5_filter_on_2d_and_3d_input():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 6, 3))
    kw = dict(fcutoff_band=np.array([3.3e6, 7.3e6]), fsampling=20.8e6, type="gaussian", order=10)
    three = RFfilter(signals=x, **kw).signal
    two = RFfilter(signals=x[:, :, 1], **kw).signal
    assert three.shape == x.shape and two.shape == x.shape[:2]
    assert np.array_equal(three[:, :, 1], two)
