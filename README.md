[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ModularUS/dasIT/blob/main/beamform_image.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# dasIT
dasIT is a delay-and-sum beamformer for planewave ultrasound implemented in Python. dasIT allows direct interaction with its functions and the exploration of processing techniques and data inside a sandbox environment in [Google Colab](https://colab.research.google.com/github/ModularUS/dasIT/blob/main/beamform_image.ipynb).

<p align="center">
<img src="https://github.com/ModularUS/dasIT/blob/main/example_data/dasIT_xt_xz.jpg" height="400">
</p>

# License
This program is free software and licensed under the Apache License v2.0 - see the [LICENSE](https://github.com/ModularUS/dasIT/blob/main/LICENSE) file for details.


# Beamforming from explicit inputs
`dasIT.beamform.Beamformer` is a delay-and-sum core that takes everything it needs as arguments in SI units. The tutorial notebook `beamform_image.ipynb` is built on it and on `dasIT.geometry`, `dasIT.signal` and `dasIT.plot`.
```python
from dasIT.geometry import linear_array, plane_wave
from dasIT.beamform import Beamformer

bf = Beamformer(element_x=linear_array(192, 0.23e-3), grid_x=x, grid_z=z, fs=20.8332e6, c=1540,
                tx=plane_wave(angle_deg=0), t0=0.0, f_number=1.5, window='hann', interp='linear', engine='matrix')
image = bf(rf)                # rf: [samples, channels] or [samples, channels, frames], real or analytic
bf.delays(element=96)         # arrival sample of every pixel on one element, bf.weights(element=96) its weights
```
`engine='matrix'` keeps the whole beamformer as one sparse matrix (fast, many frames in one product), `engine='loop'` adds element after element (a few megabytes). Both give the same image. Measured with `python tests/benchmark.py` (Hann window, linear interpolation, double precision, one frame):

| case | matrix: build | matrix: apply | matrix size | loop: apply | loop: peak memory |
|---|---|---|---|---|---|
| 192 elements, 708 x 192 pixels | 1.0 s | 0.05 s | 117 MB | 0.8 s | 7 MB |
| 128 elements, 780 x 128 pixels | 0.8 s | 0.04 s | 75 MB | 0.4 s | 6 MB |
| 128 elements, 780 x 509 pixels | 3.2 s | 0.13 s | 301 MB | 2.3 s | 23 MB |

Three ways to use it:
```python
# 1. one plane wave: rf [samples, channels]
image = Beamformer(x, grid_x, grid_z, fs, c, tx=plane_wave(0))(rf)

# 2. several steered plane waves, compounded coherently: rf [samples, channels, shots]
from dasIT.geometry import plane_waves
image = Beamformer(x, grid_x, grid_z, fs, c, tx=plane_waves([-12, 0, 12], x))(rf)

# 3. lines the scanner already beamformed: no Beamformer, only envelope, log compression and the axes
from dasIT.geometry import depth_axis, line_positions
from dasIT.signal import analytic, envelope, log_compress
from dasIT.plot import show_image
show_image(log_compress(envelope(analytic(lines)), 50), line_positions(x, lines.shape[1]), depth_axis(lines.shape[0], fs, c))
```
Channel data that were recorded one receive element per shot with the same plane wave (as the Ultrasonix `planewave_channels` sequence does) are case 1: the shots together form one `[samples, channels]` record.

# Install and test
```
pip install git+https://github.com/ModularUS/dasIT          # as a package
pip install -e .[test]                                      # from a checkout, for development
pytest
```
The tests check the beamformer against synthetic point targets with known positions (`tests/test_truth.py`), several transmits and scanner lines (`tests/test_shots.py`), the signal steps (`tests/test_signal.py`), the bundled phantom frame against a stored reference and the nominal values of the phantom (`tests/test_example.py`), and they execute the code cells of `beamform_image.ipynb`, which have to arrive at the same frame. They run on every push (`.github/workflows/tests.yml`).

Tested with Python 3.10, NumPy 1.23, SciPy 1.10 and with Python 3.12, NumPy 2.5, SciPy 1.18.

What changed against dasIT 1.x: [CHANGES.md](CHANGES.md).
