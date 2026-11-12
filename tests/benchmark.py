"""Time and memory of dasIT.beamform, both engines, on synthetic data of realistic size. A script, not a test.

    python tests/benchmark.py

build   making the sparse matrix (once per geometry and record length)
apply   one frame
peak    largest memory in use during the step (numpy allocations)
"""
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasIT.beamform import Beamformer
from dasIT.geometry import linear_array, plane_wave

CASES = [   # name, elements, pitch [m], fs [Hz], samples, lateral pixels per element, axial pixels
    ("example: 192 el, 708 x 192 px", 192, 0.23e-3, 20.8332e6, 1416, 1, 708),
    ("USX: 128 el, 780 x 128 px", 128, 0.30e-3, 20e6, 1600, 1, 780),
    ("USX, lateral grid x4: 780 x 509 px", 128, 0.30e-3, 20e6, 1600, 4, 780),
]


def measure(fn):
    tracemalloc.start()
    t = time.perf_counter()
    out = fn()
    seconds = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return out, seconds, peak


def main():
    rng = np.random.default_rng(0)
    print(f"{'case':38s} {'settings':22s} {'build':>8s} {'peak':>8s} {'matrix':>8s} {'apply':>8s} {'peak':>8s}   {'loop':>7s} {'peak':>7s}")
    for name, n_el, pitch, fs, n_samples, fine, nz in CASES:
        elements = linear_array(n_el, pitch)
        grid_x = np.linspace(elements[0], elements[-1], fine * (n_el - 1) + 1)
        grid_z = np.arange(nz) * 1540 / (2 * fs) * (n_samples / 2 / nz if nz * 2 < n_samples else 1) + 1e-4
        rf = rng.standard_normal((n_samples, n_el)) + 1j * rng.standard_normal((n_samples, n_el))
        for window, interp, dtype in (("box", "nearest", np.float64), ("hann", "linear", np.float64), ("hann", "linear", np.float32)):
            kw = dict(fs=fs, c=1540, tx=plane_wave(0), f_number=1.5, window=window, interp=interp, dtype=dtype)
            bf = Beamformer(elements, grid_x, grid_z, engine="matrix", **kw)
            data = rf.astype(np.complex64 if dtype is np.float32 else np.complex128)
            m, t_build, p_build = measure(lambda: bf.matrix(n_samples))
            size = (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e6
            a, t_apply, p_apply = measure(lambda: bf(data))
            lp = Beamformer(elements, grid_x, grid_z, engine="loop", **kw)
            b, t_loop, p_loop = measure(lambda: lp(data))
            assert np.abs(a - b).max() <= (1e-4 if dtype is np.float32 else 1e-10) * np.abs(a).max()
            label = f"{window}, {interp}, {np.dtype(dtype).name}"
            print(f"{name:38s} {label:22s} {t_build:7.2f}s {p_build:6.0f}MB {size:6.0f}MB {t_apply:7.3f}s {p_apply:6.0f}MB   {t_loop:6.2f}s {p_loop:5.0f}MB")


if __name__ == "__main__":
    main()
