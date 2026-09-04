"""Cached bootstrap, session state and shared helpers for the UI.

Everything here assumes the core packages are installed (it is imported only
after ``main.py`` passes the dependency gate).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src import spreadsheet_helpers as sh
from src.config import load_config
from src.data_loader import SIM_TEXT_COL, load_dataframe
from src.fill_down_engine import FillDownEngine
from src.ml_classifier import SETFIT_PIP_HINT, ModelManager, setfit_available
from src.rules_manager import RulesManager
from src.similarity import PIP_INSTALL_HINT, semantic_backend_available
from utils.logging_setup import configure_logging, get_logger
from utils.storage import Storage

ML_MODE_LABELS = {
    "auto": "Auto — let the tool decide (recommended)",
    "similarity_only": "Similarity only — no AI model",
    "hybrid": "Hybrid — AI and similarity work together",
    "prefer_ml": "Prefer AI — trust the trained model first",
}

friendly_engine = sh.friendly_engine


# --------------------------------------------------------------------------- #
# Bootstrap (cached so it runs once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource
def bootstrap():
    """Load config, configure logging, open storage + model manager once.

    ``FILLDOWN_DB_PATH`` overrides the SQLite location — handy for an isolated
    demo database or for headless tests so the shipped ``data/fill_down.db`` is
    never polluted.
    """
    import os

    from src.config import is_demo
    from utils import demo_utils

    config = load_config()
    db_override = os.environ.get("FILLDOWN_DB_PATH")
    if db_override:
        config.storage.db_path = db_override
    configure_logging(
        level=config.logging.level,
        json_logs=config.logging.json_logs,
        log_file=str(config.abs_log_file()),
    )
    storage = Storage(config.abs_db_path())
    model_manager = ModelManager(config, storage)
    # Fresh Demo Mode: on a deployed Space, wipe any prior data once per start so
    # the app always appears completely unused (no-op for local/production).
    demo_utils.maybe_auto_reset(config, storage, model_manager)
    rules = RulesManager(storage)
    # In demo mode start with zero rules so it looks brand new; otherwise seed
    # the illustrative starter rules as usual.
    if not is_demo():
        rules.seed_default_rules()
    _log_engine_status(config, storage, model_manager)
    return config, storage, rules, model_manager


def _log_engine_status(config, storage, model_manager) -> None:
    """Print a clear, one-time console banner of what actually loaded.

    Always succeeds — the core (TF-IDF + LogReg) path needs nothing optional, so
    we report the optional engines' real availability without ever failing.
    """
    log = get_logger("startup")
    try:
        sem_ok = semantic_backend_available()[0]
    except Exception:  # noqa: BLE001
        sem_ok = False
    try:
        sf_ok = setfit_available()[0]
    except Exception:  # noqa: BLE001
        sf_ok = False
    matching = "semantic (sentence-transformers)" if sem_ok else "TF-IDF (built-in fallback)"
    print("=" * 64)
    print(f"  Code_Down_ML v{config.app.version}")
    print(f"  DB: {config.abs_db_path()}")
    print(f"  Matching engine : {matching}")
    print(f"  Semantic AI     : {'available' if sem_ok else 'not installed (using fallback)'}")
    print(f"  SetFit model    : {'available' if sf_ok else 'not installed (LogReg only)'}")
    print(f"  Learned examples: {storage.count_training_data()}")
    print("=" * 64)
    log.info("engine_status", semantic=sem_ok, setfit=sf_ok,
             examples=storage.count_training_data())


def services():
    """Return (config, storage, rules_manager, model_manager, logger)."""
    config, storage, rules, mm = bootstrap()
    return config, storage, rules, mm, get_logger("ui")


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state(config) -> None:
    defaults = {
        "view": "landing",            # landing | spreadsheet
        "panel": None,                # None | rules | models | history
        "loaded": None,               # LoadedData
        "work_df": None,              # the authoritative working dataframe
        "original_bytes": None,
        "original_ext": None,
        "last_result": None,          # EngineResult (for breakdown / summary)
        "ml_mode": config.ml.mode,
        "undo_stack": [],
        "redo_stack": [],
        "page": 0,
        "page_size": 1500,
        "filter_mode": "All",
        "conf_filter": "All",
        "search_query": "",
        "search_scope": "All columns",
        "engine_filter": [],
        "_view_sig": None,
        "rule_panel_open": False,
        "rule_prefill": {"keyword": "", "notes": "", "code": "", "fields": []},
        "rule_prompt": None,
        "dismissed_rule_prompts": [],
        "hide_rule_suggestions": False,
        "memory_bootstrap_dismissed": False,
        "ingest_banner_dismissed": False,
        "flash": None,
        "show_install_page": False,
        "confirm_action": None,       # None | clear_spreadsheet | start_fresh
        "demo_sample_loaded": False,  # demo: auto-load sample once per session
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def goto_view(view: str, panel=None) -> None:
    """Navigation callback (use as ``on_click``) — avoids mid-run ``st.rerun``."""
    st.session_state["view"] = view
    st.session_state["panel"] = panel


def set_flash(message: str, icon: str = "") -> None:
    st.session_state["flash"] = (message, icon)


def show_flash() -> None:
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash[0])


# --------------------------------------------------------------------------- #
# Engine + loading
# --------------------------------------------------------------------------- #
def current_client_id() -> Optional[str]:
    """The active client/project name as a storage client id (None = shared)."""
    name = st.session_state.get("client_name") or ""
    return name.strip() or None


def make_engine() -> FillDownEngine:
    config, storage, rules, mm, _ = services()
    client_id = current_client_id()
    return FillDownEngine(
        config, rules,
        learned_lookup=storage.get_learned_lookup(client_id=client_id),
        model_manager=mm,
        mode=st.session_state.get("ml_mode", config.ml.mode),
        client_id=client_id,
        blocked_lookup=storage.get_blocked_lookup(client_id=client_id),
    )


def make_hybrid_engine() -> FillDownEngine:
    """Engine for the hybrid rules run: keyword rules + semantic similarity.

    Deliberately excludes the ML model (``mode="similarity_only"``) so a rules
    run is predictable — it spreads the codes users just defined via rules to
    similar transactions, using learned memory + semantic similarity only.
    """
    config, storage, rules, _mm, _ = services()
    client_id = current_client_id()
    return FillDownEngine(
        config, rules,
        learned_lookup=storage.get_learned_lookup(client_id=client_id),
        model_manager=None,
        mode="similarity_only",
        client_id=client_id,
        blocked_lookup=storage.get_blocked_lookup(client_id=client_id),
    )


def load_into_session(raw: bytes, name: str, ext: str, sheet=0) -> None:
    """Load a file, build the authoritative work_df and reset run state."""
    config, storage, *_ = services()
    loaded = load_dataframe(raw, config, source_name=name, sheet_name=sheet)
    cap = demo_max_rows()
    if cap and len(loaded.df) > cap:
        loaded.df = loaded.df.head(cap).reset_index(drop=True)
        set_flash(f"Demo limit: showing the first {cap:,} rows of this file.")
    st.session_state["loaded"] = loaded
    st.session_state["work_df"] = sh.build_work_df(loaded, storage, config)
    st.session_state["original_bytes"] = raw
    st.session_state["original_ext"] = ext
    st.session_state["last_result"] = None
    st.session_state["undo_stack"] = []
    st.session_state["redo_stack"] = []
    st.session_state["page"] = 0
    st.session_state["hide_rule_suggestions"] = False
    st.session_state["rule_prompt"] = None
    st.session_state["dismissed_rule_prompts"] = []
    st.session_state["rule_panel_open"] = False
    st.session_state["memory_bootstrap_dismissed"] = False
    st.session_state["ingest_banner_dismissed"] = False


def reset_file_session() -> None:
    """Unload the current file and clear all per-file UI state.

    Used by 'Start Fresh'. Does NOT touch persisted rules/mappings — the caller
    clears those explicitly so the action is auditable.
    """
    st.session_state["loaded"] = None
    st.session_state["work_df"] = None
    st.session_state["original_bytes"] = None
    st.session_state["original_ext"] = None
    st.session_state["last_result"] = None
    st.session_state["undo_stack"] = []
    st.session_state["redo_stack"] = []
    st.session_state["page"] = 0
    st.session_state["search_query"] = ""
    st.session_state["search_scope"] = "All columns"
    st.session_state["engine_filter"] = []
    st.session_state["filter_mode"] = "All"
    st.session_state["conf_filter"] = "All"
    st.session_state["_view_sig"] = None
    st.session_state["rule_panel_open"] = False
    st.session_state["rule_prompt"] = None
    st.session_state["dismissed_rule_prompts"] = []
    st.session_state["hide_rule_suggestions"] = False
    st.session_state["memory_bootstrap_dismissed"] = False
    st.session_state["ingest_banner_dismissed"] = False
    st.session_state["export_open"] = False
    st.session_state["panel"] = None
    st.session_state["confirm_action"] = None
    st.session_state["view"] = "landing"


def make_sample_bytes() -> bytes:
    from utils.sample_data import generate

    df = generate()
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def load_sample(navigate: bool = True) -> None:
    try:
        load_into_session(make_sample_bytes(), "sample_transactions.xlsx", ".xlsx")
        set_flash("Sample dataset loaded.")
        if navigate:
            st.session_state["view"] = "spreadsheet"
            st.session_state["panel"] = None
    except Exception as exc:  # noqa: BLE001
        set_flash(f"Could not create sample data: {exc}")


# --------------------------------------------------------------------------- #
# Undo / redo
# --------------------------------------------------------------------------- #
_UNDO_LIMIT = 30


def push_undo() -> None:
    """Snapshot the working dataframe before a mutating action."""
    work = st.session_state.get("work_df")
    if work is None:
        return
    stack = st.session_state["undo_stack"]
    stack.append(sh.snapshot(work))
    if len(stack) > _UNDO_LIMIT:
        del stack[0]
    st.session_state["redo_stack"] = []  # any new action invalidates redo


def undo() -> bool:
    work = st.session_state.get("work_df")
    stack = st.session_state["undo_stack"]
    if work is None or not stack:
        return False
    st.session_state["redo_stack"].append(sh.snapshot(work))
    sh.restore(work, stack.pop())
    return True


def redo() -> bool:
    work = st.session_state.get("work_df")
    stack = st.session_state["redo_stack"]
    if work is None or not stack:
        return False
    st.session_state["undo_stack"].append(sh.snapshot(work))
    sh.restore(work, stack.pop())
    return True


# --------------------------------------------------------------------------- #
# Demo mode (Hugging Face Spaces public demo)
# --------------------------------------------------------------------------- #
def demo_mode() -> bool:
    """True when running as the public demo (``FILLDOWN_DEMO=1`` or on HF)."""
    from src.config import is_demo
    return is_demo()


def _persistent_storage() -> bool:
    import os
    val = os.environ.get("FILLDOWN_DATA_DIR") or os.environ.get("HF_DATA_DIR")
    return bool(val) and str(val).startswith("/data")


def render_demo_banner() -> None:
    """Prominent, one-line demo notice shown on every page in demo mode."""
    if not demo_mode():
        return
    if _persistent_storage():
        st.info(
            "**Live demo** — this is a public demo of **Code_Down_ML**. "
            "Persistent storage is on, so rules and learning "
            "carry across sessions.",
            icon=":material/info:")
    else:
        st.warning(
            "**Live demo** — this is a public demo of **Code_Down_ML**. "
            "Data is not permanently saved across sessions "
            "unless persistent storage is enabled.",
            icon=":material/info:")


def reset_demo_data() -> None:
    """Manual 'Reset Demo Data' callback — wipe all data, back to brand-new."""
    from utils import demo_utils
    config, storage, rules, mm, _ = services()
    demo_utils.force_reset(config, storage, mm)
    reset_file_session()
    set_flash("Demo data cleared — the app is back to a brand-new state.")


def demo_max_rows() -> int:
    """Soft row cap for the demo (0 = unlimited). Set ``FILLDOWN_MAX_ROWS``."""
    import os
    try:
        return int(os.environ.get("FILLDOWN_MAX_ROWS", "0") or "0")
    except ValueError:
        return 0


# --------------------------------------------------------------------------- #
# Status banners / optional add-ons
# --------------------------------------------------------------------------- #
def render_backend_banner(verbose: bool = False) -> None:
    config, *_ = services()
    available, _ = semantic_backend_available()
    if available and config.similarity.use_embeddings:
        if verbose:
            st.caption("Matching engine: semantic AI (highest accuracy).")
        return
    if not config.similarity.use_embeddings:
        if verbose:
            st.caption("Matching engine: fast TF-IDF (selected in Settings).")
        return
    st.info(
        "Running on the built-in TF-IDF matcher. Optional semantic matching "
        "(higher accuracy on ambiguous wording) can be enabled from the sidebar.")


def render_optional_setup() -> None:
    sem_ok, _ = semantic_backend_available()
    sf_ok, _ = setfit_available()
    if sem_ok and sf_ok:
        return
    with st.expander("Optional engines (not required)", expanded=False):
        st.write(
            "The app runs fully on the built-in TF-IDF + LogReg engines. The "
            "optional engines below improve accuracy on ambiguous transactions.")
        if st.button("Install optional engines", type="primary",
                     key="optional_setup_install"):
            st.session_state["show_install_page"] = True
            st.rerun()
        if not sem_ok:
            st.markdown("**Semantic matching** — matches on meaning rather than "
                        "spelling (e.g. *CloudHost* ~ *Cloud Hosting*).")
            st.code(PIP_INSTALL_HINT, language="bash")
        if not sf_ok:
            st.markdown("**SetFit** — the most accurate model the tool can train "
                        "on your approvals.")
            st.code(SETFIT_PIP_HINT, language="bash")


# --------------------------------------------------------------------------- #
# Export summary table
# --------------------------------------------------------------------------- #
def export_summary_df(counts: dict, loaded, backend: str, mode: str) -> pd.DataFrame:
    rows = [
        ("File", loaded.source_name),
        ("Client", st.session_state.get("client_name") or "—"),
        ("Total transactions", counts["total"]),
        ("Already coded (examples)", counts["seeds"]),
        ("Filled automatically", counts["auto_filled"]),
        ("Filled but flagged for review", counts["filled_review"]),
        ("Need review (left blank)", counts["needs_review"]),
        ("No match (left blank)", counts["no_match"]),
        ("Total filled", counts["filled"]),
        ("Still blank", counts["blank"]),
        ("Matching engine", backend or "—"),
        ("Decision mode", mode or "—"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).astype(str)


# --------------------------------------------------------------------------- #
# Global CSS (sticky toolbar + tighter spacing)
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    """Minimal, theme-aware styling: tighter spacing, sticky toolbar, subtle panels.

    No heavy graphics — just small adjustments on top of the basic dark theme so
    the spreadsheet stays the focus and the layout is clean.
    """
    st.markdown(
        """
        <style>
        /* Tight top room so the spreadsheet dominates the viewport. */
        .block-container { padding-top: 1.3rem; padding-bottom: 1.2rem;
            max-width: 100%; }

        /* Clean, tighter typography. */
        h1, h2, h3 { font-weight: 650; letter-spacing: -0.01em; }
        h1 { font-size: 1.7rem; }
        h2 { font-size: 1.3rem; }
        h3 { font-size: 1.02rem; margin-bottom: 0.1rem; }

        /* Sticky toolbar wrapper — adapts to the active theme background. */
        div[data-testid="stVerticalBlock"] > div:has(> div.pc-toolbar) {
            position: sticky; top: 0; z-index: 999;
            background: var(--background-color);
            padding: 0.35rem 0 0.3rem 0;
            border-bottom: 1px solid rgba(128,128,128,0.22);
        }
        .pc-toolbar { font-weight: 600; }

        /* Compact vertical rhythm so more grid is visible. */
        div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
        div[data-testid="stElementContainer"]:has(> div.pc-toolbar) { margin: 0; }

        /* The data editor is the star — give it a subtle frame. */
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
            border: 1px solid rgba(128,128,128,0.22); border-radius: 6px;
        }

        /* Buttons: simple, restrained. */
        .stButton > button { border-radius: 6px; font-weight: 550; }

        /* Subtle accent panels (left border, faint background). */
        .pc-incentive, .pc-rule-prompt, .pc-seed-banner {
            border-left: 3px solid var(--primary-color, #2b8cff);
            padding: 0.55rem 0.9rem; margin: 0.3rem 0 0.5rem 0;
            background: rgba(43,140,255,0.08);
            border-radius: 4px; font-size: 0.9rem; line-height: 1.45;
            color: var(--text-color);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def incentive(message: str) -> None:
    """Subtle, professional data-value note (no hype, no emoji)."""
    st.markdown(f"<div class='pc-incentive'>{message}</div>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Value messaging (time saved + client insight)
# --------------------------------------------------------------------------- #
# Conservative estimate of the hands-on time a reviewer would spend looking up
# and keying one transaction code manually. Deliberately modest so the figure
# stays credible.
SECONDS_SAVED_PER_TXN = 15


def _format_span(minutes: int) -> str:
    if minutes >= 60:
        h, m = divmod(minutes, 60)
        return f"{h}h {m}m" if m else f"{h}h"
    return f"{minutes} min"


def minutes_saved(counts: dict) -> int:
    """Estimated minutes saved from automatically coded transactions."""
    secs = int(counts.get("auto_filled", 0)) * SECONDS_SAVED_PER_TXN
    return int(round(secs / 60))


def value_note(counts: dict) -> str:
    """A subtle, professional one-liner about time saved + client insight.

    Returns an empty string before anything is auto-coded so the UI stays clean.
    """
    parts = []
    mins = minutes_saved(counts)
    if mins > 0:
        parts.append(
            f"≈ {_format_span(mins)} of manual coding saved "
            f"({counts.get('auto_filled', 0):,} auto-coded)")
    accts = int(counts.get("distinct_accounts", 0))
    if accts:
        parts.append(f"across {accts:,} account"
                     f"{'s' if accts != 1 else ''}")
    return " · ".join(parts)
