"""Dependency detection + in-app installer (standard-library only).

This module is intentionally lightweight: it imports **nothing** beyond the
Python standard library so it can run on a half-installed machine (e.g. before
``pandas`` or ``scikit-learn`` exist). The Streamlit UI uses it to:

* decide whether the app can run at all (core deps), and whether the optional
  semantic / ML stack is present;
* install missing dependencies from within the app via ``pip`` in a subprocess,
  streaming the output back to the UI in real time.

Nothing here ever raises on a missing optional package — callers get a clear
status object instead.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Tuple

# Project root = one level above this file's package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Requirements files shipped alongside the app.
CORE_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
SEMANTIC_REQUIREMENTS = PROJECT_ROOT / "requirements-semantic.txt"

# --------------------------------------------------------------------------- #
# Dependency groups
#
# Each entry is (pip_name, import_name). ``pip_name`` is what we hand to pip;
# ``import_name`` is the top-level module we probe with importlib.
# --------------------------------------------------------------------------- #

CORE_PACKAGES: List[Tuple[str, str]] = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("openpyxl", "openpyxl"),
    ("streamlit", "streamlit"),
    ("pydantic", "pydantic"),
    ("PyYAML", "yaml"),
    ("structlog", "structlog"),
    ("scikit-learn", "sklearn"),
]

SEMANTIC_PACKAGES: List[Tuple[str, str]] = [
    ("sentence-transformers", "sentence_transformers"),
    ("torch", "torch"),
    ("setfit", "setfit"),
]

# CPU-only wheel index for torch so non-technical users don't pull multi-GB
# CUDA builds they can't use. Used by the "Install Full" path.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


@dataclass
class GroupStatus:
    """Availability of a named dependency group."""

    name: str
    present: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    @property
    def total(self) -> int:
        return len(self.present) + len(self.missing)


@dataclass
class DependencyStatus:
    """Snapshot of which dependency groups are installed."""

    core: GroupStatus
    semantic: GroupStatus

    @property
    def core_ok(self) -> bool:
        return self.core.ok

    @property
    def semantic_ok(self) -> bool:
        return self.semantic.ok

    @property
    def core_missing(self) -> List[str]:
        return self.core.missing

    @property
    def semantic_missing(self) -> List[str]:
        return self.semantic.missing


def _module_available(import_name: str) -> bool:
    """True if ``import_name`` can be found without importing it.

    ``find_spec`` does not execute the module, so this stays fast and never
    triggers a heavy ``torch`` import.
    """
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _check_group(name: str, packages: List[Tuple[str, str]]) -> GroupStatus:
    present: List[str] = []
    missing: List[str] = []
    for pip_name, import_name in packages:
        if _module_available(import_name):
            present.append(pip_name)
        else:
            missing.append(pip_name)
    return GroupStatus(name=name, present=present, missing=missing)


def check_dependencies() -> DependencyStatus:
    """Return a fresh snapshot of core + semantic dependency availability.

    ``invalidate_caches`` ensures packages installed during this session (via the
    in-app installer) are discovered on the next rerun without a full restart.
    """
    importlib.invalidate_caches()
    return DependencyStatus(
        core=_check_group("core", CORE_PACKAGES),
        semantic=_check_group("semantic", SEMANTIC_PACKAGES),
    )


# --------------------------------------------------------------------------- #
# Installation (streamed via pip subprocess)
# --------------------------------------------------------------------------- #

# Sentinel emitted as the final streamed line so the UI knows the exit code.
EXIT_SENTINEL = "__PIP_EXIT__"


def _stream_command(cmd: List[str]) -> Iterator[str]:
    """Run ``cmd`` and yield its combined stdout/stderr line by line.

    The final yielded line is ``f"{EXIT_SENTINEL} {returncode}"`` so callers can
    detect success (0) or failure (non-zero) without a second call.
    """
    yield f"$ {' '.join(cmd)}"
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
    except Exception as exc:  # noqa: BLE001 - surface launch failures to the UI
        yield f"Failed to start pip: {exc}"
        yield f"{EXIT_SENTINEL} 1"
        return

    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()
    yield f"{EXIT_SENTINEL} {proc.returncode}"


# Version floors (mirrors requirements*.txt) applied only when a package is
# actually missing, so we never churn / upgrade already-working packages.
_VERSION_SPEC = {
    "pandas": "pandas>=2.1.0",
    "numpy": "numpy>=1.26.0",
    "openpyxl": "openpyxl>=3.1.2",
    "streamlit": "streamlit>=1.32.0",
    "pydantic": "pydantic>=2.6.0",
    "PyYAML": "PyYAML>=6.0.1",
    "structlog": "structlog>=24.1.0",
    "scikit-learn": "scikit-learn>=1.4.0",
    "sentence-transformers": "sentence-transformers>=2.5.0",
    "torch": "torch>=2.5.0",
    "setfit": "setfit>=1.0.0",
}

# transformers 5.x dropped APIs that SetFit still uses; pin to the 4.x line. This
# is appended to semantic installs so a clean machine ends up with a working
# SetFit (pip would otherwise pull the latest, incompatible, transformers).
TRANSFORMERS_PIN = "transformers>=4.41,<5"


def _spec(pip_name: str) -> str:
    return _VERSION_SPEC.get(pip_name, pip_name)


def install_core() -> Iterator[str]:
    """Stream installation of *only the missing* core packages.

    Crucially we never pass ``--upgrade`` and never reinstall packages that are
    already present. That keeps the running Streamlit / numpy / pandas untouched
    (reinstalling the live Streamlit would fail with a file-lock on Windows and
    can corrupt the environment).
    """
    missing = check_dependencies().core_missing
    if not missing:
        yield "All core packages are already installed — nothing to do."
        yield f"{EXIT_SENTINEL} 0"
        return
    yield f"Installing missing core packages: {', '.join(missing)}"
    yield from _stream_command(
        [sys.executable, "-m", "pip", "install", "--no-warn-script-location",
         *[_spec(name) for name in missing]]
    )


def install_semantic(cpu_only: bool = True) -> Iterator[str]:
    """Stream installation of *only the missing* optional semantic / ML packages.

    When ``cpu_only`` is set (the default, recommended), torch
    is pulled from the CPU wheel index to avoid huge CUDA downloads. Only missing
    packages are installed, so nothing already working is upgraded or disturbed.
    """
    missing = check_dependencies().semantic_missing
    if not missing:
        yield "All semantic + ML packages are already installed — nothing to do."
        yield f"{EXIT_SENTINEL} 0"
        return
    yield f"Installing missing packages: {', '.join(missing)}"
    specs = [_spec(name) for name in missing]
    # Always enforce the transformers pin so SetFit/sentence-transformers stay
    # compatible (cheap no-op if a compatible version is already installed).
    specs.append(TRANSFORMERS_PIN)
    cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location",
           *specs]
    if cpu_only and "torch" in missing:
        cmd += ["--extra-index-url", TORCH_CPU_INDEX]
    yield from _stream_command(cmd)


def parse_exit_code(line: str) -> int | None:
    """Return the exit code if ``line`` is the sentinel, else ``None``."""
    if line.startswith(EXIT_SENTINEL):
        try:
            return int(line.split(maxsplit=1)[1])
        except (IndexError, ValueError):
            return 1
    return None
