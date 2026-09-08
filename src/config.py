"""Typed configuration loaded from ``config.yaml``.

The whole config is validated through Pydantic so a malformed YAML file fails
loudly and early with a helpful message, rather than blowing up deep inside the
pipeline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

__version__ = "1.1.0"

# Project root = the directory that contains config.yaml (one level above src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _env_flag(name: str) -> bool:
    """True when ``name`` is set to 1 / true / yes (case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def is_hf_space() -> bool:
    """True when running inside a Hugging Face Space (HF sets ``SPACE_ID``)."""
    return bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))


def is_demo() -> bool:
    """True for the public demo build.

    Triggered by ``FILLDOWN_DEMO=1`` (set in the Dockerfile) or automatically
    when deployed to a Hugging Face Space. Detection is independent of
    ``is_live()`` — when both flags are set, live *wins* on wipe / sample
    auto-load / auth (see ``demo_reset_enabled`` and ``ui.common.demo_mode``).
    """
    return _env_flag("FILLDOWN_DEMO") or is_hf_space()


def is_live() -> bool:
    """True for a hosted client instance (``FILLDOWN_LIVE=1``).

    Railway is treated as live only when this flag is set — ``is_railway()``
    alone never flips a public demo into live mode.
    """
    return _env_flag("FILLDOWN_LIVE")


def is_railway() -> bool:
    """True when Railway injects its environment / volume variables."""
    return bool(os.environ.get("RAILWAY_ENVIRONMENT")
                or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH"))


def auth_password() -> str:
    """Shared password for the live gate. Empty means 'not set'."""
    return os.environ.get("FILLDOWN_AUTH_PASSWORD", "").strip()


def live_boot_blocked() -> bool:
    """True when live mode must refuse to render (password missing).

    Local and public-demo runs are never blocked. The password is never
    logged or returned here — callers only learn whether boot is allowed.
    """
    return is_live() and not bool(auth_password())


def demo_reset_enabled() -> bool:
    """Whether Fresh Demo Mode should auto-wipe data on startup.

    On by default in demo/HF context; set ``FILLDOWN_DEMO_RESET=0`` to keep data
    across restarts (useful with HF persistent storage). Never true off-demo,
    and **never true in live mode** even if ``FILLDOWN_DEMO=1`` is also set —
    a hosted client book must survive deploys and restarts.
    """
    if is_live():
        return False
    if not is_demo():
        return False
    return os.environ.get("FILLDOWN_DEMO_RESET", "1").strip().lower() \
        not in {"0", "false", "no"}


class DataDirUnwritable(RuntimeError):
    """Live instance cannot write the configured data directory.

    Raised instead of silently falling through to ``/tmp/code_down_ML``, which
    would look like persistence is on while every deploy wipes the book.
    """


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:  # noqa: BLE001
        return False


def _configured_data_dirs() -> List[Path]:
    dirs: List[Path] = []
    for env in ("FILLDOWN_DATA_DIR", "HF_DATA_DIR"):
        val = os.environ.get(env)
        if val:
            dirs.append(Path(val).expanduser())
    return dirs


def select_data_dir() -> Path:
    """Pick the first writable data directory.

    Honours ``FILLDOWN_DATA_DIR`` / ``HF_DATA_DIR`` (Hugging Face persistent
    storage mounts at ``/data``). Falls back to the project-local ``data/`` and
    finally a temp dir, so the app never crashes on a read-only filesystem
    (e.g. an HF Space without persistent storage attached).

    In **live** mode the silent temp fallback is forbidden: if the configured
    dir (or ``./data`` when unset) is not writable, raise
    :class:`DataDirUnwritable` so the operator sees it.
    """
    env_dirs = _configured_data_dirs()
    if is_live():
        preferred = env_dirs[0] if env_dirs else (PROJECT_ROOT / "data")
        # Prefer the first writable configured dir; never walk into /tmp.
        for cand in (env_dirs or [preferred]):
            if _is_writable(cand):
                return cand
        target = env_dirs[0] if env_dirs else preferred
        raise DataDirUnwritable(
            "This hosted client instance cannot write its data directory "
            f"({target}). Mount a persistent volume at /data and set "
            "RAILWAY_RUN_UID=0 so the container can write it. Refusing to "
            "fall back to a temporary directory — that would look like "
            "persistence is on while every restart wipes the book."
        )

    candidates = list(env_dirs)
    candidates.append(PROJECT_ROOT / "data")
    candidates.append(Path(tempfile.gettempdir()) / "code_down_ML")
    for cand in candidates:
        if _is_writable(cand):
            return cand
    # Last resort: project-local (config.ensure_dirs will surface any error).
    return PROJECT_ROOT / "data"


def probe_live_data_dir() -> Optional[str]:
    """Return an error message if the live data dir is unusable, else None."""
    if not is_live():
        return None
    try:
        select_data_dir()
    except DataDirUnwritable as exc:
        return str(exc)
    return None


class AppConfig(BaseModel):
    name: str = "ProfitCoach"
    version: str = __version__


class SimilarityConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.72
    cluster_eps: Optional[float] = None
    min_cluster_size: int = 2
    use_embeddings: bool = True
    cache_embeddings: bool = True
    # Strip long numeric tokens (check #s, dates, amounts) from similarity text.
    drop_numeric_tokens: bool = True
    # Collapse identical field values so repeated text doesn't dominate.
    dedupe_fields: bool = True

    @property
    def eps(self) -> float:
        """DBSCAN epsilon in cosine-distance space."""
        if self.cluster_eps is not None:
            return self.cluster_eps
        return max(0.01, 1.0 - self.similarity_threshold)


class ConfidenceConfig(BaseModel):
    auto_apply_cutoff: float = 0.85
    review_cutoff: float = 0.55
    rule_match_confidence: float = 0.99
    learned_match_confidence: float = 0.97


class MLConfig(BaseModel):
    """Trainable ML layer that sits *on top* of the similarity engine.

    The whole layer is optional: when disabled (or when no model has been
    trained, or when dependencies are missing) the engine falls back to pure
    similarity and behaves exactly as it always has.
    """

    enabled: bool = True
    # auto | similarity_only | hybrid | prefer_ml
    # "auto" picks the behaviour progressively from how many labelled examples
    # have been collected (see hybrid_min / ml_primary_min below).
    mode: str = "auto"
    # An ML prediction must reach this confidence to be trusted as "high".
    ml_confidence_cutoff: float = 0.85
    # SetFit base model (kept small for fast CPU training).
    setfit_model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    setfit_num_epochs: int = 1
    setfit_batch_size: int = 16
    # Progressive thresholds (labelled-example counts).
    hybrid_min: int = 300
    ml_primary_min: int = 1500
    # Training guards.
    min_examples_to_train: int = 10
    test_size: float = 0.2
    # Where versioned trained models live (relative to project root).
    model_store_dir: str = "data/models"
    # Offer a "train now" reminder after this many new approvals since the last
    # successful train (0 = never). Training itself stays user-triggered.
    auto_train_every: int = 25


class ColumnsConfig(BaseModel):
    new_account: List[str] = Field(default_factory=lambda: ["New Account"])
    text_columns: List[str] = Field(default_factory=list)
    amount: List[str] = Field(default_factory=lambda: ["Amount"])
    date: List[str] = Field(default_factory=lambda: ["Date"])
    # Vendor-first: Name and Memo carry the strongest signal on real ledgers.
    similarity_columns: List[str] = Field(
        default_factory=lambda: ["Name", "Memo", "Description", "Account",
                                 "Category"]
    )
    # Preference order for mining keyword-rule suggestions (vendor first).
    keyword_source: List[str] = Field(
        default_factory=lambda: ["Name", "Memo", "Description", "Payee",
                                 "Split", "Account", "Category"]
    )
    # Never mined for keyword suggestions unless explicitly opted into via
    # keyword_source. Any header containing "note" is also excluded.
    keyword_source_exclude: List[str] = Field(
        default_factory=lambda: ["Notes", "Note", "Internal Notes",
                                 "Rule Notes"]
    )


class StorageConfig(BaseModel):
    db_path: str = "data/fill_down.db"
    work_dir: str = "data/work"


class LoggingConfig(BaseModel):
    # populate_by_name lets us keep the friendly YAML key ``json`` while using
    # ``json_logs`` internally (avoids shadowing BaseModel.json()).
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_logs: bool = Field(default=False, alias="json")
    log_file: str = "data/fill_down.log"


class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Optional NARPM-style glossary: canonical code -> {"name": ..., "keywords":
    # [...]}. Enriches rationales and rule suggestions; empty = disabled.
    account_glossary: Dict[str, object] = Field(default_factory=dict)

    # Resolved absolute paths (filled in during load).
    project_root: str = str(PROJECT_ROOT)
    # Writable root for all runtime artifacts (DB, models, work, logs). On HF
    # Spaces this is /data (persistent storage); locally it is ``<root>/data``.
    data_dir: str = str(PROJECT_ROOT / "data")

    # ----------------------------------------------------------- behaviour
    def similarity_columns_effective(self) -> List[str]:
        """The columns currently used to build similarity text.

        Kept as a method so the UI can override ``columns.similarity_columns``
        live and every consumer immediately picks up the change.
        """
        return list(self.columns.similarity_columns)

    def keyword_source_order(self) -> List[str]:
        """Preference order for mining keyword-rule suggestions."""
        return list(self.columns.keyword_source)

    def keyword_source_excluded(self) -> set:
        """Lowercased headers never mined for keyword suggestions."""
        return {str(c).strip().lower() for c in self.columns.keyword_source_exclude}

    def glossary_entry(self, code: object) -> Dict[str, object]:
        """Glossary entry for an account code (``{}`` when unknown/disabled).

        Lookup is normalised (``6100 a`` finds the ``6100A`` entry) and never
        raises on a malformed glossary — an empty/bad glossary simply disables
        the enrichment.
        """
        if not self.account_glossary:
            return {}
        try:
            from utils.account_codes import normalize_code
            key = normalize_code(code)
            for raw_key, entry in self.account_glossary.items():
                if normalize_code(raw_key) == key and isinstance(entry, dict):
                    return entry
        except Exception:  # noqa: BLE001 - glossary is best-effort
            return {}
        return {}

    # ------------------------------------------------------------------ paths
    def abs_db_path(self) -> Path:
        return self._resolve(self.storage.db_path)

    def abs_work_dir(self) -> Path:
        return self._resolve(self.storage.work_dir)

    def abs_log_file(self) -> Path:
        return self._resolve(self.logging.log_file)

    def abs_model_store_dir(self) -> Path:
        return self._resolve(self.ml.model_store_dir)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        # Relative ``data/...`` paths live under the (possibly relocated) data
        # dir so HF persistent storage and local runs both work unchanged.
        parts = path.parts
        if parts and parts[0] == "data":
            base = Path(self.data_dir)
            return base.joinpath(*parts[1:]) if len(parts) > 1 else base
        return Path(self.project_root) / path

    def abs_sessions_dir(self) -> Path:
        """Per-client live workspace folders (``{data_dir}/sessions/<slug>/``)."""
        return Path(self.data_dir) / "sessions"

    def ensure_dirs(self) -> None:
        """Create any directories the app needs to write into."""
        for path in (self.abs_db_path().parent, self.abs_work_dir(),
                     self.abs_log_file().parent, self.abs_model_store_dir(),
                     self.abs_sessions_dir()):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: Optional[os.PathLike | str] = None) -> Config:
    """Load and validate configuration from YAML.

    Falls back to sensible defaults if the file is missing so the app still
    runs out of the box. Raises a clear error on malformed YAML.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: Dict = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"config.yaml is not valid YAML: {exc}"
            ) from exc

    config = Config(**data)
    config.project_root = str(PROJECT_ROOT)
    config.data_dir = str(select_data_dir())
    config.ensure_dirs()
    return config
