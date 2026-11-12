"""The golden generations: from what dasIT 1.0 computes to the corrected chain, one change at a time.

    python tests/generations.py            recompute all generations, print the table, write tests/golden/current/

Every generation is the previous one with ONE thing changed (CHANGES below). All of them beamform frame 0 of the
example with dasIT.beamform.Beamformer; what differs is the input signal and the settings. Generation v1 is the
frozen legacy result tests/golden/usdatarecycler and is not computed here.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from dasIT import signal as ds
from dasIT.beamform import Beamformer
from dasIT.data.loader import load_rf
from dasIT.geometry import linear_array, plane_wave
from legacy_formula import core_settings, legacy_geometry
from phantom import measure

EXAMPLE = HERE.parent / "example_data" / "CIRSphantom_GE9LD_VVantage"
LEGACY, CURRENT = HERE / "golden" / "usdatarecycler", HERE / "golden" / "current"
PITCH, N_ELEMENTS, C, START_SAMPLES, CROP = 0.23e-3, 192, 1540.0, 40, 1416
BAND_HZ = (3312500.0, 7287500.0)                       # the band edges of transducer.csv
# From the Verasonics setup of the recording (Trans.frequency, Receive.startDepth, Trans.lensCorrection, P.endDepth),
# all in wavelengths at 5.2083 MHz and 1540 m/s:
FC, START_DEPTH_WL, LENS_WL, DESIGN_DEPTH_WL = 5208300.0, 5.0, 3.3820129870129874, 95.0
T0 = 2 * (START_DEPTH_WL - LENS_WL) / FC               # s: the record starts 2 x 5 periods after the transmit, the lens
                                                       # delays every echo by 2 x 3.38 periods. +0.62 us, 0.48 mm in depth
DESIGN_DEPTH_M = DESIGN_DEPTH_WL * 1540.0 / FC         # 28.1 mm: the depth the acquisition was set up for
F_NUMBER, FINE = 1.5, 3                                # default aperture; lateral pixels per element of the display grid
OVERSAMPLE = 4                                         # of the record before linear interpolation: 1.2 % error against 8x (1x: 21 %, 2x: 5.5 %)

CHANGES = {
    "v2": "a delay beyond the record adds nothing; path lengths in double precision",
    "v3": "the full record of 4096 samples instead of the crop to 1416",
    "v4": "element positions one pitch apart (were stretched by 192/191)",
    "v5": "aperture z / 1.5 with a Hann window that uses every element inside it",
    "v6": "linear interpolation between samples, after 4x oversampling of the record",
    "v7": "every pixel divided by the sum of its weights",
    "v8": "a real band-pass filter (Butterworth, zero phase) instead of the 10-tap filter that passed DC",
    "v9": "no gain curve applied (v1.0 multiplied by the raw TGC control values)",
    "v10": "three lateral pixels per element, beamformed (not interpolated afterwards)",
    "v11": "the first 40 samples kept (v1.0 set them to zero; the record starts at 5 wavelengths depth, they are echoes)",
    "v12": "time of sample 0 from the acquisition settings: start depth minus lens, +0.62 us",
}


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


# ---- the signal before beamforming
def raw_record(keep_start=False):
    """Frame 0, [4096 samples, 192 channels], the samples before the first echo set to zero as in the notebook."""
    rf = load_rf(EXAMPLE / "CIRS_phantom.h5")[:, :, 0, 0]
    if not keep_start:
        rf[:START_SAMPLES] = 0
    return rf


def legacy_gain(n_samples):
    """The v1.0 gain: the raw control values as factors, interpolated over the cropped record, constant after it."""
    points = np.loadtxt(EXAMPLE / "tgc_cntrl_pt.csv", delimiter=",")
    curve = np.interp(np.arange(CROP), np.arange(0, CROP, CROP // points.size), points)
    return np.concatenate([curve, np.full(max(0, n_samples - CROP), curve[-1])])[:n_samples]


def legacy_signal(n_samples):
    """Analytic signal as v1.0 makes it (gain, 10-tap filter, Hilbert), on a record of n_samples."""
    rf = raw_record()[:n_samples] * legacy_gain(n_samples)[:, None]
    taps = np.load(LEGACY / "golden.npz")["filter_coefficients"]
    return ds.analytic(sps.convolve(rf, taps[:, None], mode="same"))


def corrected_signal(gain, keep_start=False):
    rf = raw_record(keep_start)
    if gain:
        rf = rf * legacy_gain(rf.shape[0])[:, None]
    return ds.analytic(ds.bandpass(rf, BAND_HZ, 4 * 5208300.0))


# ---- the generations
def settings(name):
    """(analytic signal, fs, Beamformer keyword arguments) of a generation."""
    g = legacy_geometry()
    k = int(name[1:])
    true_x = linear_array(N_ELEMENTS, PITCH)
    signal = legacy_signal(CROP if k == 2 else 4096) if k <= 7 else corrected_signal(gain=(k == 8), keep_start=(k >= 11))
    fs = g["fs"]
    kw = dict(core_settings(g, "hann"), interp="nearest")
    if k >= 4:                                                    # true positions; the legacy element count per depth kept
        kw.update(element_x=true_x, grid_x=true_x, half_aperture=(g["active"] - 1) / 2 * PITCH)
    if k >= 5:
        kw.update(half_aperture=None, f_number=F_NUMBER)
    if k >= 6:
        signal, fs = ds.oversample(signal, OVERSAMPLE), OVERSAMPLE * fs
        kw.update(interp="linear", fs=fs)
    if k >= 7:
        kw.update(normalise=True)
    if k >= 12:
        kw.update(t0=T0)
    if k >= 10:
        kw.update(grid_x=np.linspace(true_x[0], true_x[-1], FINE * (N_ELEMENTS - 1) + 1))
    return signal, kw


def measured(name, frame, x, z):
    """The phantom metrics of a generation. From v12 on the image is drawn c T0 / 2 deeper."""
    shift = 1e3 * 1540.0 * T0 / 2 if name != "v1" and int(name[1:]) >= 12 else 0.0
    return measure(np.abs(frame), x, z, depth_shift_mm=shift)


def generation(name):
    """(beamformed frame [depth, lateral], grid_x, grid_z) of a generation v2 and later."""
    signal, kw = settings(name)
    return Beamformer(tx=plane_wave(0), **kw)(signal), kw["grid_x"], kw["grid_z"]


def legacy_v1():
    g = legacy_geometry()
    return np.load(LEGACY / "golden.npz")["beamformed"].astype(np.complex128), g["x"], g["z"]


def compare(new, old):
    """How a generation differs from its parent (same grid): size of the change and the rows it touches."""
    if new.shape != old.shape:
        return None
    d = np.abs(new - old)
    rows = np.flatnonzero(d.max(axis=1) > 1e-9 * np.abs(old).max())
    return dict(max_over_max=float(d.max() / np.abs(old).max()), mean_over_mean=float(d.mean() / np.abs(old).mean()),
                rows_changed=int(rows.size), first_row_changed=int(rows[0]) if rows.size else None,
                envelope_correlation=float(np.corrcoef(np.abs(new).ravel(), np.abs(old).ravel())[0, 1]))


def lineage():
    """All generations: hash, change against the parent, phantom metrics. Returns (list of entries, last frame)."""
    import scipy
    parent, x, z = legacy_v1()
    entries = [dict(name="v1", change="dasIT 1.0 / usdatarecycler as frozen in tests/golden/usdatarecycler", sha256=None,
                    shape=list(parent.shape), against_parent=None, phantom=measured("v1", parent, x, z))]
    for name in CHANGES:
        frame, x, z = generation(name)
        entries.append(dict(name=name, change=CHANGES[name], sha256=sha(frame), shape=list(frame.shape),
                            against_parent=compare(frame, parent), phantom=measured(name, frame, x, z)))
        parent = frame
    versions = dict(numpy=np.__version__, scipy=scipy.__version__)
    return dict(versions=versions, generations=entries), (frame, x, z)


def table(entries):
    rows = ["| gen | change | lateral scale | depth scale | depth offset | wire width at 20 mm (lat x ax) | cylinder contrast | speckle | change vs parent (mean) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        p, a = e["phantom"], e["against_parent"]
        rows.append("| %s | %s | %+.2f %% | %+.2f %% | %+.2f mm | %.2f x %.2f mm | %.1f dB | %.2f | %s |" % (
            e["name"], e["change"], p["lateral_scale_percent"], p["depth_scale_percent"], p["depth_offset_mm"], p["wire_width_lateral_mm"][1],
            p["wire_width_axial_mm"][1], p["cylinder_contrast_db"], p["speckle_snr"], "%.3g %%" % (100 * a["mean_over_mean"]) if a else "other grid" if e["name"] != "v1" else ""))
    return "\n".join(rows)


LINEAGE_HEAD = """# Lineage of the golden model

Generated by `python tests/generations.py`. Do not edit by hand.

Each generation is its parent with one change. `v1` is what dasIT 1.0 / usdatarecycler computes, frozen in
`tests/golden/usdatarecycler/`. The newest generation is stored in `tests/golden/current/golden.npz`, the hash and
the measurements of every generation in `tests/golden/current/lineage.json`. `tests/test_corrections.py` recomputes
all of them.

The columns are measurements on the CIRS 054GS phantom of the example frame (`tests/phantom.py`). Nominal values:
lateral scale 0 %, depth scale 0 %, depth offset 0 mm, speckle 1.91.

"""
LINEAGE_OPEN = """

## Open items

- **Depth scale -4.5 % in every generation.** It does not come from the beamformer (v1 and v12 agree) and not from
  the time of the first sample (an offset, corrected in v12). The Verasonics setup of the recording confirms the
  sampling rate (NS200BW at 5.2083 MHz) and 1540 m/s.
  A sampling rate of 20.0 MHz would bring the wire spacing to 19.9 mm, but it focuses the wires worse than 20.83 MHz,
  and the acquisition system derives its sampling rates from a 250 MHz clock (20.83 MHz = 250 / 12), so 20.0 MHz is
  not a rate it offers. The cause is still open.
- **The wires are about 1.0 mm wide laterally, theory for f/1.5 at 5.2 MHz is about 0.64 mm.** Unexplained. Likely
  linked to the item above.
- **Depth offset.** Since v12 the time of sample 0 follows the acquisition settings. The near-field wires then sit
  about 1.2 mm deeper than their nominal depth below the phantom surface. A layer of coupling gel between probe and
  phantom would do that; it was not measured.
- **The image was designed for 28 mm depth** (P.endDepth = 95 wavelengths). Deeper pixels are formed from the part of
  their aperture whose echoes still fall into the record.
- **Transmit angle.** The acquisition fired three plane waves per frame (-12, 0, +12 degrees). The example file holds
  one shot per frame, the 0 degree transmit according to its author. The data alone cannot tell 0 from -12 degrees.
- **v5 widens the wires from 0.92 to 1.05 mm.** Expected: the aperture changes from z / 1.27 (the old rule, taken from
  the elevation focus) to z / 1.5. The f-number is a setting; 1.5 is the default, not a measured optimum for this probe.
"""


def write(result, frame, x, z):
    CURRENT.mkdir(parents=True, exist_ok=True)
    (CURRENT / "lineage.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    np.savez_compressed(CURRENT / "golden.npz", beamformed=frame.astype(np.complex64), grid_x=x, grid_z=z)
    (HERE / "golden" / "LINEAGE.md").write_text(LINEAGE_HEAD + table(result["generations"]) + LINEAGE_OPEN, encoding="utf-8")


if __name__ == "__main__":
    result, (frame, x, z) = lineage()
    write(result, frame, x, z)
    print(table(result["generations"]))
