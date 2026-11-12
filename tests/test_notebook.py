"""The code cells of beamform_image.ipynb run against this checkout, top to bottom, in one namespace.

Lines that start a shell command (!pip) are left out; the download cell finds the data bundled in example_data/ and
does nothing. Figures are drawn off screen. The notebook has to arrive at the newest golden generation
(tests/golden/current, see tests/golden/LINEAGE.md) and at its wire widths.
"""
import contextlib
import io
import json
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "beamform_image.ipynb"
CURRENT = ROOT / "tests" / "golden" / "current"
pytestmark = pytest.mark.skipif(not (ROOT / "example_data" / "CIRSphantom_GE9LD_VVantage" / "CIRS_phantom.h5").exists(),
                                reason="example data not present")


def code_cells(path):
    """(cell number, source without shell lines) of every code cell."""
    for number, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.strip():
            yield number, "\n".join(line for line in source.split("\n") if not line.lstrip().startswith("!"))


@pytest.fixture(scope="module")
def namespace():
    here = os.getcwd()
    os.chdir(ROOT)                                            # the notebook reads example_data/... relative to the root
    matplotlib.use("Agg", force=True)
    names, failed = {}, []
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            for number, source in code_cells(NOTEBOOK):
                try:
                    exec(compile(source, f"{NOTEBOOK.name} cell {number}", "exec"), names)
                except Exception as error:                    # noqa: BLE001
                    failed.append(f"cell {number}: {type(error).__name__}: {error}")
    finally:
        os.chdir(here)
        matplotlib.pyplot.close("all")
    assert not failed, "\n".join(failed)
    return names


def test_notebook_arrives_at_the_golden_frame(namespace):
    golden = np.load(CURRENT / "golden.npz")["beamformed"]
    image = namespace["image"]
    assert image.shape == golden.shape
    assert np.abs(image - golden).max() <= 1e-6 * np.abs(golden).max()


def test_notebook_measures_the_wire_widths_of_the_lineage(namespace):
    newest = json.loads((CURRENT / "lineage.json").read_text(encoding="utf-8"))["generations"][-1]["phantom"]
    assert namespace["lateral_mm"] == pytest.approx(newest["wire_width_lateral_mm"][1], abs=0.1)      # the wire at 20 mm
    assert namespace["axial_mm"] == pytest.approx(newest["wire_width_axial_mm"][1], abs=0.1)


def test_notebook_compounding_lowers_the_side_lobes(namespace):
    x, row = namespace["sim_x"], namespace["row"]
    outside = np.abs(x - namespace["points"][0][0]) > 1.5e-3
    level = lambda image: 20 * np.log10(image[row][outside].max() / image[row].max())
    assert level(namespace["compound"]) < level(namespace["single"]) - 3
