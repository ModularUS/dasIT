"""The example frame: unchanged against the stored reference, and physically plausible on the phantom.

E1  the frame equals tests/golden/golden.npz
E2  the phantom measurements equal tests/golden/metrics.json
E3  nominal values of the CIRS 054GS phantom that the image has to meet
"""
import json
from pathlib import Path

import numpy as np
import pytest

from example import DATA, example_frame
from phantom import measure

GOLDEN = Path(__file__).resolve().parent / "golden"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="example data not present")


@pytest.fixture(scope="module")
def frame():
    return example_frame()


@pytest.fixture(scope="module")
def measured(frame):
    image, x, z = frame
    return measure(np.abs(image), x, z)


def test_e1_frame_is_the_reference(frame):
    image, x, z = frame
    stored = np.load(GOLDEN / "golden.npz")
    assert image.shape == stored["beamformed"].shape
    assert np.array_equal(x, stored["grid_x"]) and np.allclose(z, stored["grid_z"], rtol=0, atol=1e-12)
    assert np.abs(image - stored["beamformed"]).max() <= 2e-6 * np.abs(stored["beamformed"]).max()


def test_e2_phantom_measurements_are_the_reference(measured):
    stored = json.loads((GOLDEN / "metrics.json").read_text(encoding="utf-8"))["phantom"]
    for key, value in stored.items():
        assert measured[key] == pytest.approx(value, abs=0.02), key


def test_e3_phantom_nominal_values(measured):
    assert abs(measured["lateral_scale_percent"]) < 0.2            # the near-field wires are 6 mm apart
    assert measured["near_depth_step_mm"] == pytest.approx(1.0, abs=0.1)     # and 1 mm apart in depth
    assert measured["wire_width_axial_mm"][1] < 0.5                # the wire at 20 mm: a few periods of 0.3 mm wavelength
    assert measured["cylinder_contrast_db"] > 6                    # the hyperechoic cylinder stands out
    assert measured["speckle_snr"] == pytest.approx(1.91, abs=0.2)  # fully developed speckle
