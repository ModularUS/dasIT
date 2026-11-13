"""Write the reference of the example frame from the code as it is now.

    python tests/golden/make_golden.py

golden.npz     the beamformed frame of tests/example.py (complex64) and its pixel grid
metrics.json   the phantom measurements on it (tests/phantom.py) and the library versions

Run it only when a change of the result is intended, and say in the commit what changed and why.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
from example import example_frame  # noqa: E402
from phantom import measure  # noqa: E402

frame, x, z = example_frame()
np.savez_compressed(HERE / "golden.npz", beamformed=frame.astype(np.complex64), grid_x=x, grid_z=z)
metrics = dict(versions=dict(numpy=np.__version__, scipy=scipy.__version__), phantom=measure(np.abs(frame), x, z))
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
print(json.dumps(metrics["phantom"], indent=1))
