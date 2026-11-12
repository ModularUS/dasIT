"""The code cells of beamform_image.ipynb run against this checkout, top to bottom, in one namespace.

Lines that start a shell command (!pip, !wget) and the cell that downloads the example data on Colab are left out:
the test uses the data bundled in example_data/. Figures are drawn off screen.
"""
import contextlib
import io
import json
import warnings
from pathlib import Path

import matplotlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "beamform_image.ipynb"
pytestmark = pytest.mark.skipif(not (ROOT / "example_data" / "CIRSphantom_GE9LD_VVantage" / "CIRS_phantom.h5").exists(),
                                reason="example data not present")


def code_cells(path):
    """(cell number, source without shell lines) of every code cell, without the Colab download cell."""
    for number, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or not source.strip() or "/content/rfdata" in source:
            continue
        yield number, "\n".join(line for line in source.split("\n") if not line.lstrip().startswith("!"))


def test_notebook_cells_run(monkeypatch):
    monkeypatch.chdir(ROOT)                                   # the notebook reads example_data/... relative to the root
    matplotlib.use("Agg", force=True)
    namespace, failed = {}, []
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        for number, source in code_cells(NOTEBOOK):
            try:
                exec(compile(source, f"{NOTEBOOK.name} cell {number}", "exec"), namespace)
            except Exception as error:                        # noqa: BLE001
                failed.append(f"cell {number}: {type(error).__name__}: {error}")
    matplotlib.pyplot.close("all")
    assert not failed, "\n".join(failed)
    assert namespace["BFsignals"].frame.shape == (708, 192)   # the notebook did beamform the example frame
