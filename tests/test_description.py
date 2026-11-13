"""The beamforming description: a recording described by one dictionary gives the right image.

P1  a point reflector appears at its position, for one and for several transmits, with a time of sample 0
P2  the description survives JSON, and equals a Beamformer set up by hand
P3  prepare: silent reference, blanked start of channels, band-pass
P4  an override changes one setting and nothing else
P5  a wrong description is refused with the name of the key
"""
import numpy as np
import pytest

from dasIT import description as dd
from dasIT.beamform import Beamformer
from dasIT.geometry import linear_array, plane_wave
from dasIT.signal import analytic, envelope
from synthetic import point_echoes

C, FC, FS, N = 1540.0, 5.0e6, 40.0e6, 2000
ELEMENTS = linear_array(64, 0.3e-3)
POINT = (2.1e-3, 14e-3)


def describe(angles=(0.0,), t0=0.0, **more):
    return dict(element_x=ELEMENTS.tolist(), fs=FS, c=C, t0=t0, transmits=[dict(angle_deg=a, x0=0.0) for a in angles], **more)


def record(angles=(0.0,), t0=0.0):
    """[samples, channels(, shots)] of the point; sample 0 is taken at time t0."""
    shots = [point_echoes([POINT], ELEMENTS, FS, N, C, plane_wave(a), FC, t0=t0).real for a in angles]
    return shots[0] if len(shots) == 1 else np.stack(shots, axis=2)


@pytest.mark.parametrize("angles,t0", [((0.0,), 0.0), ((0.0,), 1.5e-6), ((-8.0, 0.0, 8.0), 0.0)])
def test_p1_point_at_its_position(angles, t0):
    d = describe(angles, t0)
    rf = analytic(record(angles, t0))
    bf = dd.beamformer(d, n_samples=N)
    image = envelope(bf(rf))
    iz, ix = np.unravel_index(image.argmax(), image.shape)
    assert abs(bf.grid_x[ix] - POINT[0]) <= 0.3e-3 and abs(bf.grid_z[iz] - POINT[1]) <= 0.1e-3


def test_p2_json_and_hand_made_beamformer_agree():
    d = describe(f_number=1.2, window="hann")
    again = dd.loads(dd.dumps(d))
    assert again == dd.complete(d)
    rf = analytic(record())
    by_hand = Beamformer(ELEMENTS, ELEMENTS, dd.beamformer(d, n_samples=N).grid_z, FS, C, tx=plane_wave(0), f_number=1.2, window="hann")
    assert np.allclose(dd.beamformer(again, n_samples=N)(rf), by_hand(rf), rtol=0, atol=1e-12)


def test_p3_prepare():
    rng = np.random.default_rng(0)
    pickup = rng.standard_normal((N, ELEMENTS.size))
    rf = record() + pickup + 3.0                                   # echoes + pickup + an offset
    skip = np.zeros(ELEMENTS.size, int)
    skip[2::4] = 40
    d = describe(band_hz=[3e6, 7e6], skip_samples=skip.tolist())
    out = dd.prepare(rf, d, silent=pickup)
    clean = dd.prepare(record(), describe(band_hz=[3e6, 7e6]))
    assert np.abs(out[200:-200] - clean[200:-200]).max() < 1e-6 * np.abs(clean).max()      # pickup and offset are gone
    assert rf.dtype == np.float64 and np.all(rf[0] != 0)           # the input is left alone
    raw = dd.prepare(rf, describe(skip_samples=skip.tolist()))     # no band: only the blanking
    assert np.all(raw[:40, 2::4] == 0) and np.all(raw[:40, 0::4] != 0) and np.array_equal(raw[40:], rf[40:])
    frames = np.stack([rf, rf], axis=2)
    assert dd.prepare(frames, d, silent=pickup).shape == frames.shape


def test_p3_shots_written_side_by_side():
    angles = (-8.0, 0.0, 8.0)
    d = describe(angles)
    stacked = record(angles)                                        # [samples, channels, shots]
    side_by_side = np.concatenate([stacked[:, :, k] for k in range(3)], axis=1)       # [samples, shots * channels]
    assert np.array_equal(dd.prepare(side_by_side, d), dd.prepare(stacked, d))
    frames = np.stack([side_by_side, 2 * side_by_side], axis=2)
    out = dd.prepare(frames, d)
    assert out.shape == (N, ELEMENTS.size, 3, 2) and np.array_equal(out[..., 1], 2 * out[..., 0])
    image = envelope(dd.beamformer(d, n_samples=N)(analytic(dd.prepare(side_by_side, d))))
    iz, ix = np.unravel_index(image.argmax(), image.shape)
    assert abs(ELEMENTS[ix] - POINT[0]) <= 0.3e-3


def test_p4_override_changes_one_setting():
    d = describe()
    wide, narrow = dd.beamformer(d, n_samples=N), dd.beamformer(d, n_samples=N, f_number=3.0)
    assert np.count_nonzero(wide.weights(32)) > np.count_nonzero(narrow.weights(32))
    assert np.array_equal(wide.grid_z, narrow.grid_z) and dd.complete(d)["f_number"] == 1.5          # the description itself is unchanged
    faster = dd.beamformer(d, n_samples=N, c=1600.0)
    assert faster.grid_z[-1] == pytest.approx(wide.grid_z[-1] * 1600.0 / C)


@pytest.mark.parametrize("change,word", [(dict(fs=-1), "fs"), (dict(band_hz=[3e6, 30e6]), "band_hz"), (dict(window="gauss"), "window"),
                                         (dict(transmits=[]), "transmits"), (dict(skip_samples=[1, 2]), "skip_samples"), (dict(bandwidth=1), "bandwidth")])
def test_p5_wrong_description_is_refused(change, word):
    with pytest.raises(ValueError, match=word):
        dd.complete(dict(describe(), **change))


def test_p5_missing_key_and_channel_count():
    d = describe()
    del d["c"]
    with pytest.raises(ValueError, match="'c'"):
        dd.complete(d)
    with pytest.raises(ValueError, match="channels"):
        dd.prepare(np.zeros((N, 10)), describe())
    with pytest.raises(ValueError, match="n_samples"):
        dd.beamformer(describe())
