"""The minimal sidebar: navigation, engine status and advanced settings."""

from __future__ import annotations

import streamlit as st

from src import dependencies as deps
from src.data_loader import SIM_TEXT_COL
from src.similarity import semantic_backend_available
from ui import setup
from ui.common import services


def _goto_view(view: str) -> None:
    st.session_state["view"] = view
    st.session_state["panel"] = None


def _open_panel(panel: str) -> None:
    # Open a management modal without leaving the current base view. Only one
    # modal may be open per run, so close any export modal first.
    st.session_state["panel"] = panel
    st.session_state["export_open"] = False


def _nav_button(label: str, view: str, *, key: str, disabled=False) -> None:
    active = (st.session_state.get("view") == view
             and not st.session_state.get("panel"))
    st.button(label, width="stretch", key=key, disabled=disabled,
              type="primary" if active else "secondary",
              on_click=_goto_view, args=(view,))


def _panel_button(label: str, panel: str, *, key: str) -> None:
    st.button(label, width="stretch", key=key, on_click=_open_panel,
              args=(panel,))


def render_sidebar() -> None:
    config, storage, rules, mm, _ = services()
    has_data = st.session_state.get("work_df") is not None

    with st.sidebar:
        st.title("📊 Code_Down_ML")
        st.caption(f"Transaction classification · v{config.app.version}")

        st.markdown("#### Workspace")
        _nav_button("Spreadsheet", "spreadsheet",
                    key="nav_sheet", disabled=not has_data)
        _nav_button("Review", "review", key="nav_review", disabled=not has_data)
        _nav_button("New file", "landing", key="nav_landing")

        st.markdown("#### Manage")
        _panel_button("Rules", "rules", key="nav_rules")
        _panel_button("Models", "models", key="nav_models")
        _panel_button("History", "history", key="nav_history")

        st.divider()
        setup.render_sidebar_dependencies(deps.check_dependencies())

        with st.expander("Advanced tuning", expanded=False):
            config.similarity.similarity_threshold = st.slider(
                "Similarity threshold", 0.30, 0.99,
                float(config.similarity.similarity_threshold), 0.01,
                help="Higher values require transactions to look more alike.")
            config.confidence.auto_apply_cutoff = st.slider(
                "Auto-fill confidence", 0.50, 0.99,
                float(config.confidence.auto_apply_cutoff), 0.01,
                help="Fill automatically only at or above this confidence.")
            config.confidence.review_cutoff = st.slider(
                "Send-to-review threshold", 0.10, 0.95,
                float(config.confidence.review_cutoff), 0.01,
                help="Below this confidence, flag for review instead of filling.")

            loaded = st.session_state.get("loaded")
            if loaded is not None:
                options = [c for c in loaded.df.columns if c != SIM_TEXT_COL]
                default = [c for c in config.columns.similarity_columns
                           if c in options] or loaded.text_columns
                chosen = st.multiselect(
                    "Transaction description columns", options=options,
                    default=default,
                    help="Typically Description, Name and Memo. Applies next run.")
                if chosen:
                    config.columns.similarity_columns = chosen

        available, _ = semantic_backend_available()
        config.similarity.use_embeddings = st.toggle(
            "Semantic (AI) matching",
            value=config.similarity.use_embeddings and available,
            disabled=not available,
            help=("Matches on meaning rather than spelling." if available else
                  "Optional engine not installed — the TF-IDF matcher is used."))

        from ui import common as _common
        scoped_mm = _common.model_manager_for(_common.current_client_id())
        n_examples = scoped_mm.training_count()
        learning_on = config.ml.enabled and (mm.has_model()
                                             or scoped_mm.has_model())
        st.caption(f"Learning: {'on' if learning_on else 'warming up'} · "
                   f"{n_examples} examples")
        st.divider()
        st.caption("Theme: top-right menu → Settings → Appearance.")
