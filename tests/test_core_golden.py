"""The golden frames connected to the new core (dasIT.beamform), link by link. No old defect is rebuilt as an option
of the new code: every link is either exact or a counted, explained difference (tests/legacy_formula.py).

    golden frame  ==  formula as the old code computes                        exact (to the single precision it is stored in)
       link 0         main only: the box was applied to the delay index, so elements outside the aperture added sample 0
       link A         path lengths in double instead of single precision: 644 delays round to the other neighbour
       link B         a delay beyond the record adds nothing, the old code added sample 0 of that channel
    clean formula ==  new core with the legacy geometry and aperture           exact, for both engines

The numbers of the links are properties of the frozen golden frames. They are asserted here so that a change in
any of them is noticed.
"""
from pathlib import Path

import numpy as np
import pytest

from dasIT.beamform import Beamformer
from dasIT.geometry import plane_wave
from legacy_formula import core_settings, legacy_delays, legacy_frame, legacy_geometry, legacy_weights

GOLDEN = Path(__file__).parent / "golden"
CASES = {   # branch: window, sample limit, box applied on the index, [delays changed by link A, entries of link B]
    "usdatarecycler": dict(window="hann", limit=1415, mask_on_index=False, link_a=644, link_b=188890),
    "main": dict(window="box", limit=1360, mask_on_index=True, link_a=644, link_b=900620),
}


def rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def mean_rel(a, b):
    return float(np.abs(a - b).mean() / np.abs(b).mean())


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    c = dict(CASES[request.param], branch=request.param, g=legacy_geometry())
    stored = np.load(GOLDEN / request.param / "golden.npz")
    c["golden"], c["analytic"] = stored["beamformed"], stored["analytic"].astype(np.complex128)
    frame = lambda paths, read0, on_index=False: legacy_frame(c["analytic"], c["g"], c["window"], c["limit"], paths, read0, on_index)
    c["as_old_code"] = frame(np.float32, True, c["mask_on_index"])
    c["weights_on_signal"] = frame(np.float32, True)
    c["double_paths"] = frame(np.float64, True)
    c["clean"] = frame(np.float64, False)
    return c


def test_formula_is_the_golden_frame(case):
    assert rel(case["as_old_code"], case["golden"]) < 2e-7


def test_link_0_box_on_the_index(case):
    if not case["mask_on_index"]:
        pytest.skip("usdatarecycler applies the window to the signal")
    assert 3e-4 < rel(case["weights_on_signal"], case["as_old_code"]) < 5e-4
    assert 0.009 < mean_rel(case["weights_on_signal"], case["as_old_code"]) < 0.012


def test_link_a_double_precision_paths(case):
    g, inside = case["g"], legacy_weights(case["g"], case["window"]) != 0
    changed = (legacy_delays(g, np.float32) != legacy_delays(g, np.float64)) & inside
    assert changed.sum() == case["link_a"]
    assert rel(case["double_paths"], case["weights_on_signal"]) < 2e-2
    assert mean_rel(case["double_paths"], case["weights_on_signal"]) < 1e-3


def test_link_b_nothing_beyond_the_record(case):
    g, inside = case["g"], legacy_weights(case["g"], case["window"]) != 0
    beyond = (legacy_delays(g, np.float64) > case["limit"]) & inside
    assert beyond.sum() == case["link_b"]
    assert rel(case["clean"], case["double_paths"]) < 1e-3
    assert mean_rel(case["clean"], case["double_paths"]) < 1e-3
    rows = np.flatnonzero(beyond.any(axis=(1, 2)))
    untouched = np.setdiff1d(np.arange(len(g["z"])), rows)
    assert np.array_equal(case["clean"][untouched], case["double_paths"][untouched])     # confined to the deepest rows


@pytest.mark.parametrize("engine", ["matrix", "loop"])
def test_new_core_equals_the_clean_formula(case, engine):
    record = case["analytic"][:case["limit"] + 1]             # the old depth limit as the length of the record
    bf = Beamformer(tx=plane_wave(0), interp="nearest", engine=engine, **core_settings(case["g"], case["window"]))
    assert rel(bf(record), case["clean"]) < 1e-12
