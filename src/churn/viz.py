"""Plotting helpers shared by the Unidad 2 and 3 scripts.

The original scripts saved figures to a hardcoded/pre-existing folder
(a relative "unidad 2/" that had to already exist, or in one case an
absolute path from a teammate's own machine). ``output_dir`` replaces both:
it always resolves relative to the repo root and creates the folder itself.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def output_dir(unit: str) -> Path:
    """Return outputs/<unit>/, creating it if needed."""
    path = REPO_ROOT / "outputs" / unit
    path.mkdir(parents=True, exist_ok=True)
    return path
