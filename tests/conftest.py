"""Makes the repository root importable, so the tests run against this checkout without an install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
