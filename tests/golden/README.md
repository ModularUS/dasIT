# Golden model

Reference results of the dasIT processing chain on the bundled example data (`example_data/CIRSphantom_GE9LD_VVantage`,
frame 0), made on 2026-11-12 with the code of each branch and the library versions the code was written for
(Python 3.10, NumPy 1.23.5, SciPy 1.10.1). A refactored dasIT has to reproduce these results, or deviate from them
for a stated reason.

| Folder | Code | Settings |
|---|---|---|
| `main/` | branch `main` at `ce07090` | `beamform_image.ipynb`, blanks filled from `transducer.csv`, apodization `rec` |
| `usdatarecycler/` | branch `origin/usdatarecycler` at `83b2393` | the same, apodization `hanning` (`rec` does not exist there) |

Each folder holds `golden.npz` (the result of every stage: rf, tgc, filtered, analytic, beamformed, envelope,
interpolated, log_image, plus one column of the delay and apodization tables and the filter coefficients) and
`golden.json` (settings, geometry, library versions, and per stage the shape, dtype, SHA-256 of the full-precision
array, statistics, run time and peak memory). `deviations_from_the_notebook` in the json lists what had to differ from
the notebook to make the chain run. On `main` that is one item: the filter call fails as written, the golden uses the
same coefficients with the kernel shape of tag v1.0.

```
python tests/golden/make_golden.py <checkout of a branch> <output folder>   # regenerate, in the old environment
python tests/golden/check_formulas.py                                       # recompute both beamformed frames from formulas
```

`check_formulas.py` reproduces both beamformed frames from closed formulas to single precision (2e-7 and 4e-8 of the
maximum). It documents what the code computes, including what is known to be wrong. It is not a statement of what the
correct image is.
