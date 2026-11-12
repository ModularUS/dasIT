"""dasIT against the beamformer of the Ultrasonix pipeline (usx.imaging.das_planewave) on recorded frames.

Needs recordings that are not part of this repository. Set the environment variable DASIT_USX_FUSED to a fused.h5
written by `ustemg export`, or to a folder that holds run folders with one each; without it the test is skipped.

A fused file carries the channel data (/us/rf: one plane wave, one receive element per shot, which makes an ordinary
[samples, channels] record) and what das_planewave made of it (/us/bf) together with the settings it used. dasIT has
to reproduce /us/bf from /us/rf with the same settings:

    band-pass of the channel data, Butterworth order 4, zero phase
    box aperture of width z / f_number, linear interpolation, time zero as stored
    one pixel per element and per sample

The single known difference is at the end of the record and is described in the test.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from dasIT import signal as ds
from dasIT.beamform import Beamformer
from dasIT.geometry import plane_wave

h5py = pytest.importorskip("h5py")
LOCATION = os.environ.get("DASIT_USX_FUSED")


def fused_files():
    if not LOCATION:
        return []
    path = Path(LOCATION)
    return [path] if path.is_file() else sorted(path.glob("*/fused.h5"))


FILES = fused_files()
pytestmark = pytest.mark.skipif(not FILES, reason="no Ultrasonix recording (set DASIT_USX_FUSED)")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.parent.name[-12:])
def test_dasit_reproduces_das_planewave(path):
    with h5py.File(path, "r") as f:
        if "us/bf" not in f:
            pytest.skip("no beamformed layer in this file")
        us, bf = f["us"].attrs, f["us/bf"]
        fs, c = float(us["sampling_frequency_hz"]), float(us["speed_of_sound_ms"])
        x, z = f["us/x_m"][:], f["us/z_m"][:]
        beamformer = Beamformer(x, x, z, fs, c, tx=plane_wave(0), t0=-float(bf.attrs["das_t0_samples"]) / fs,
                                f_number=float(bf.attrs["f_number"]), window="box", interp="linear")
        frames = sorted({0, bf.shape[0] // 2, bf.shape[0] - 1})
        # The one difference between the two: an echo that arrives between the last two samples of the record is
        # interpolated by dasIT and left out by das_planewave. It concerns the rows whose widest element (at the
        # edge of the aperture) hears the pixel that late; all rows above them have to be identical.
        f_number, n_samples = float(bf.attrs["f_number"]), bf.shape[1]
        latest = np.arange(n_samples) / 2 * (1 + np.sqrt(1 + 1 / (4 * f_number ** 2)))      # arrival sample at the aperture edge
        same = int(np.flatnonzero(latest <= n_samples - 2)[-1]) + 1
        for k in frames:
            rf = ds.bandpass(f["us/rf"][k].astype(np.float64), tuple(bf.attrs["das_band_hz"]), fs, order=4)
            mine, theirs = beamformer(rf), bf[k].astype(np.float64)
            scale = np.abs(theirs).max()
            assert np.abs(mine[:same] - theirs[:same]).max() < 5e-6 * scale, f"frame {k}"
            assert np.abs(mine[same:] - theirs[same:]).max() < 0.2 * scale, f"frame {k}, rows at the end of the record"
