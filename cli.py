"""Console entry point so the app can be launched with ``code-down``.

Installed via ``pip install -e .`` (see pyproject.toml). It simply shells out
to ``streamlit run main.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app = Path(__file__).resolve().parent / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
