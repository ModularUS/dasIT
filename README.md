[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ModularUS/dasIT/blob/main/beamform_image.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# dasIT
dasIT is a delay-and-sum beamformer for planewave ultrasound implemented in Python. dasIT allows direct interaction with its functions and the exploration of processing techniques and data inside a sandbox environment in [Google Colab](https://colab.research.google.com/github/ModularUS/dasIT/blob/main/beamform_image.ipynb).

<p align="center">
<img src="https://github.com/ModularUS/dasIT/blob/main/example_data/dasIT_xt_xz.jpg" height="400">
</p>

# License
This program is free software and licensed under the Apache License v2.0 - see the [LICENSE](https://github.com/ModularUS/dasIT/blob/main/LICENSE) file for details.


# Install and test
```
pip install git+https://github.com/ModularUS/dasIT          # as a package
pip install -e .[test]                                      # from a checkout, for development
pytest                                                      # compares every processing stage with the golden model
```
The tests run the chain of `beamform_image.ipynb` on the bundled example data and compare each stage with the reference results in `tests/golden/` (see `tests/golden/README.md`). They also execute the code cells of the notebook itself.

Tested with Python 3.10, NumPy 1.23, SciPy 1.10, pandas 1.5 (the versions dasIT 1.0 was written for) and with Python 3.12, NumPy 2.5, SciPy 1.18, pandas 3.1.
