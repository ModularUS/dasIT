# Reference of the example frame

| File | Content |
|---|---|
| `golden.npz` | the example frame beamformed by `tests/example.py` (complex64, 708 x 574) and its pixel grid |
| `metrics.json` | the measurements on the CIRS 054GS phantom in that frame (`tests/phantom.py`), and the NumPy and SciPy versions it was written with |

`tests/test_example.py` recomputes the frame and compares it with both files. `tests/test_notebook.py` runs
`beamform_image.ipynb` and requires that it arrives at the same frame.

Rewrite the reference with `python tests/golden/make_golden.py`, only when a change of the result is intended.
Say in the commit what changed and why.

The reference guards against unintended change. Whether the result is right is the job of `tests/test_truth.py`
(synthetic point targets with known positions, independent of dasIT) and of the nominal values of the phantom.
