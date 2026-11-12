"""Both golden beamformed frames recomputed from closed formulas, independent of the dasIT code.

    python tests/golden/check_formulas.py          (needs numpy only, any current version)

This is the written-down definition of what dasIT computes on each branch: delays, aperture, weights, sum.
dasIT computes the path lengths in float32, which decides the rounding of some delays, so the check does too."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
FOLDER = {"main": "main", "recycler": "usdatarecycler"}
for b in ("main", "recycler"):
    g, j = np.load(HERE / FOLDER[b] / "golden.npz"), json.loads((HERE / FOLDER[b] / "golden.json").read_text())
    geo, st = j["geometry"], j["settings"]
    fs, c, N, pitch = geo["sampling_frequency_hz"], 1540.0, st["transducer_elements_nr"], st["element_pitch_m"]
    x = np.linspace(-pitch * N / 2, pitch * N / 2, N)
    z = np.linspace(0, geo["wavelength_m"] * 177, 708, endpoint=False)
    A = g["analytic"]                                   # complex64 copy of the golden analytic signal
    tx = np.broadcast_to(z[:, None, None], (len(z), N, N)).astype(np.float32)
    rx = np.sqrt(z[:, None, None] ** 2 + (x[None, :, None] - x[None, None, :]) ** 2).astype(np.float32)
    d = np.rint(np.multiply((tx + rx) / c, fs))
    limit = 1360 if b == "main" else 1416 - 1
    d[d > limit] = 0
    d = d.astype(np.int64)
    F = 1 / (2 * ((pitch * N / 2) / st["elevation_focus"]))
    n = np.ceil((z / (2 * F)) * N / (pitch * N)) // 2 * 2 + 1
    off = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    ch = np.arange(N)[None, :, None]
    if b == "main":
        bf = A[d * (off[None] <= ((n - 1) / 2)[:, None, None]), ch].sum(axis=1)
    else:
        w = np.zeros((len(z), N, N))
        for k in np.unique(n).astype(int):
            w[n == k] = np.where(off <= (k - 1) // 2, np.hanning(k)[np.minimum((k - 1) // 2 + off, k - 1)], 0.0)
        bf = (A[d, ch] * w).sum(axis=1)
    ref = g["beamformed"]
    print(f"{b:9s} delay table centre column identical: {np.array_equal(d[:, 96, :], g['delays_centre_column'])} | "
          f"beamformed frame: max abs difference / max = {np.abs(bf - ref).max() / np.abs(ref).max():.1e} (inputs stored as complex64)")
