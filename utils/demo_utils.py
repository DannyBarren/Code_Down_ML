"""Fresh Demo Mode — wipe all user data so a deployed Space looks brand new.

Only ever runs in demo / Hugging Face context (see ``src.config.is_demo``); a
local or production install is never auto-wiped. The schema, config, sample data
loader and all core code are preserved — only *data* is cleared:

    * every row in the SQLite DB (rules, learned mappings, rule notes, training
      examples, run history) and the id counters,
    * every trained model under ``data/models/`` (general + per-client),
    * the log file and the work/temp dir contents.

After a reset the app is fully functional: users can create clients and rules,
the sample dataset still loads, and all engines still run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

from src.config import Config, demo_reset_enabled, is_demo
from utils.logging_setup import get_logger
from utils.storage import Storage

logger = get_logger("demo")

# Set once per process: a cold container start (the unit HF restarts) resets once.
_RESET_DONE = False


def _clear_dir_contents(path: Path) -> int:
    """Remove everything inside ``path`` (keep the dir itself). Returns count."""
    removed = 0
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return 0
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
            removed += 1
        except Exception:  # noqa: BLE001
            logger.warning("demo_reset_unlink_failed", path=str(child))
    return removed


def clear_model_store(config: Config) -> int:
    """Delete all trained models (and the registry). Returns items removed."""
    return _clear_dir_contents(config.abs_model_store_dir())


def clear_runtime_files(config: Config) -> None:
    """Truncate the log file and clear the work/temp dir."""
    try:
        log = config.abs_log_file()
        if log.exists():
            log.unlink()
    except Exception:  # noqa: BLE001
        logger.warning("demo_reset_log_failed")
    _clear_dir_contents(config.abs_work_dir())


def reset_for_demo(config: Config, storage: Storage,
                   model_manager=None) -> Dict[str, int]:
    """Wipe all user data. Returns a small summary of what was cleared.

    Safe to call repeatedly. Does nothing destructive to schema or code.
    """
    storage.reset_all()
    models_removed = clear_model_store(config)
    clear_runtime_files(config)
    # Drop any in-memory model cache so freshly-empty state is served.
    if model_manager is not None and hasattr(model_manager, "_cache"):
        try:
            model_manager._cache.clear()
        except Exception:  # noqa: BLE001
            pass
    logger.info("demo_reset_done", models_removed=models_removed)
    return {"models_removed": models_removed}


def maybe_auto_reset(config: Config, storage: Storage,
                     model_manager=None) -> bool:
    """Run a one-time startup reset when Fresh Demo Mode is enabled.

    Returns True if a reset happened. Guarded so it runs only once per process
    (the bootstrap that calls this is itself cached once per process).
    """
    global _RESET_DONE
    if _RESET_DONE or not demo_reset_enabled():
        return False
    reset_for_demo(config, storage, model_manager)
    _RESET_DONE = True
    logger.info("demo_auto_reset_applied")
    return True


def force_reset(config: Config, storage: Storage, model_manager=None) -> Dict[str, int]:
    """Manual 'Reset Demo Data' button — always resets (when in demo)."""
    if not is_demo():
        return {}
    return reset_for_demo(config, storage, model_manager)
