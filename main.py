"""ProfitCoach — Fill Down Automation v1.1.

Streamlit entry point (thin router).

Run with:  streamlit run main.py

This file does only four things:
    1. Guarantee the app launches even when run as ``python main.py`` or when
       heavy dependencies are missing (graceful relaunch + Setup screen).
    2. Verify the core packages, otherwise show the in-app installer.
    3. Boot the shared services and session state.
    4. Route between the two main views — **Dashboard** and **Spreadsheet** — plus
       the secondary management panels (Rules / Models / History).

All real logic lives in ``src/`` (business logic) and ``ui/`` (presentation).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force the PyTorch-only Hugging Face backend before anything imports it.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


def _running_under_streamlit() -> bool:
    """True only when launched via ``streamlit run`` (not ``python main.py``)."""
    try:
        from streamlit.runtime import exists
        if exists():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        return False


# Transparently relaunch under Streamlit if someone runs ``python main.py``.
if not _running_under_streamlit():
    import subprocess

    print("Starting Code_Down_ML…")
    print("(Tip: you can also launch with `streamlit run main.py`.)\n")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "streamlit", "run",
             str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        sys.exit(completed.returncode)
    except FileNotFoundError:
        print("Streamlit is not installed yet. Install the core dependencies "
              "first:\n    python -m pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


st.set_page_config(
    page_title="Code_Down_ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ``src.dependencies`` and ``ui.setup`` import only the standard library +
# Streamlit, so this is safe even before pandas / scikit-learn exist.
from src import dependencies as deps  # noqa: E402
from ui import setup  # noqa: E402

# --------------------------------------------------------------------------- #
# Dependency gate
# --------------------------------------------------------------------------- #
_dep_status = deps.check_dependencies()
if not _dep_status.core_ok:
    setup.render_setup_screen(_dep_status)
    st.stop()

# Live auth + data-dir probe run *before* ``common.services()`` / bootstrap.
# bootstrap is ``@st.cache_resource`` and calls ``maybe_auto_reset`` — a public
# visitor must not trigger that, and a missing volume must fail visibly.
from src.config import DataDirUnwritable, probe_live_data_dir  # noqa: E402
from ui.live_gate import enforce_live_auth  # noqa: E402

enforce_live_auth()
_live_dir_err = probe_live_data_dir()
if _live_dir_err:
    st.error(_live_dir_err)
    st.stop()

# Core deps are present — safe to import the rest of the app.
from ui import common, insights, landing, panels, review, sidebar, spreadsheet  # noqa: E402
from ui import guide  # noqa: E402  (isolated, additive User Guide page)

try:
    config, storage, rules_manager, model_manager, logger = common.services()
except DataDirUnwritable as exc:
    st.error(str(exc))
    st.stop()
common.init_state(config)
common.current_client_id()  # rehydrate durable client name after widget-key drop
common.inject_css()

# Demo convenience: in demo/HF mode, load the neutral sample dataset once per
# session so the app opens with data ready to explore (a clean pitch experience).
# Never auto-loads off-demo or in live mode, and the one-time flag means
# "New file" / "Start fresh" are respected afterwards.
if (common.should_auto_load_sample()
        and not st.session_state.get("demo_sample_loaded")):
    if st.session_state.get("work_df") is None:
        common.load_sample(navigate=True)
    st.session_state["demo_sample_loaded"] = True

# Full-page installer (reached from the sidebar) takes over when requested.
if st.session_state.get("show_install_page"):
    setup.render_install_page()
    st.stop()

sidebar.render_sidebar()

# Isolated, additive: a prominent button to open the User Guide. Appended to the
# sidebar without modifying ui/sidebar.py.
with st.sidebar:
    st.divider()
    st.button("📖 User Guide", width="stretch", type="primary",
              key="open_user_guide", on_click=guide.open_guide)
    st.caption(f"Code_Down_ML • v{config.app.version}")

# Full-page User Guide takeover (mirrors the installer screen pattern). When
# open it replaces only the main content; closing returns to the exact same
# app state. No existing view, panel or data path is touched.
if guide.is_open():
    guide.render_guide()
    st.stop()

common.render_demo_banner()
common.show_flash()

# --------------------------------------------------------------------------- #
# Router: secondary panel > spreadsheet > dashboard
# --------------------------------------------------------------------------- #
_panel = st.session_state.get("panel")
_view = st.session_state.get("view", "landing")

# Base view is always rendered; management panels overlay as modals so the
# spreadsheet is never taken over.
if _view == "spreadsheet" and st.session_state.get("work_df") is not None:
    spreadsheet.render_spreadsheet()
elif _view == "review" and st.session_state.get("work_df") is not None:
    review.render_review()
elif _view == "insights" and st.session_state.get("work_df") is not None:
    insights.render_insights()
else:
    landing.render_landing()

if _panel:
    panels.open_panel_dialog(_panel)

# Footer on the main app pages.
st.divider()
st.caption(f"© 2026 Danny Barren — Code_Down_ML v{config.app.version}")
