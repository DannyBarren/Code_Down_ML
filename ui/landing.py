"""The Dashboard view: pick the client, upload a file (or load the sample)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data_loader import DataLoadError, list_excel_sheets
from ui import common
from ui.common import services


def _render_resume_controls(client_name: str) -> None:
    """Offer Resume last workspace for the typed client, or the only saved one."""
    if client_name and common.live_workspace_exists(client_name):
        st.info("A saved workspace exists for this client.")
        if st.button("Resume last workspace", type="primary",
                     width="stretch", key="resume_named"):
            if common.restore_workspace(client_name):
                st.rerun()
            else:
                st.error("Could not restore that workspace.")
        return

    sessions = common.list_live_sessions()
    if not client_name and len(sessions) == 1:
        saved = sessions[0]
        saved_name = saved.get("client_name") or saved.get("slug") or ""
        rows = saved.get("row_count", "?")
        file_name = saved.get("source_name") or "workbook"
        st.info(
            f"One saved workspace: **{saved_name}** — {file_name} "
            f"({rows} rows). Resume restores that client name.")
        if st.button("Resume last workspace", type="primary",
                     width="stretch", key="resume_only"):
            if common.restore_workspace(saved_name):
                st.rerun()
            else:
                st.error("Could not restore that workspace.")


def render_landing() -> None:
    config, storage, rules, mm, logger = services()

    st.title("📊 Code_Down_ML")
    st.caption(f"Intelligent Transaction Classification · v{config.app.version}")

    if common.demo_mode():
        st.success(
            "**Welcome to the live demo.** Seed a few codes and the app "
            "propagates them to every similar transaction, scores its confidence, "
            "and flags anything uncertain for review — turning hours of "
            "spreadsheet work into a one-click, reviewable workflow. Click "
            "**Load sample dataset** to try it in seconds.")

    # ---- At-a-glance metric cards ---------------------------------------- #
    try:
        n_rules = len(rules.list_rules())
    except Exception:  # noqa: BLE001
        n_rules = 0
    n_examples = storage.count_training_data()
    n_runs = len(storage.list_runs(limit=1000))
    m1, m2, m3 = st.columns(3)
    m1.metric("Rules defined", f"{n_rules:,}")
    m2.metric("Examples learned", f"{n_examples:,}")
    m3.metric("Runs logged", f"{n_runs:,}")

    st.write(
        "Upload a transaction export. The application assigns the "
        "**Target Account** code for transactions that match, reports a "
        "confidence score and rationale for each, and presents everything in a "
        "single reviewable spreadsheet.")

    common.incentive(
        "Every confirmed code is retained as reusable matching data — subsequent "
        "files are classified faster and more accurately.")

    common.render_backend_banner(verbose=True)

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 1 · Client")
        st.session_state.setdefault("client_name", "")
        live = common.live_mode()
        st.text_input(
            "Client / project name (required)" if live
            else "Client / project name (optional)",
            key="client_name",
            placeholder="e.g. Northwind Trading Co.",
            help=("Required. Scopes rules, learned memory, training, models, "
                  "and the saved workspace folder. Switching the name switches "
                  "all of those."
                  if live else
                  "Used to label exports and run history."))
        client_name = (st.session_state.get("client_name") or "").strip()
        if live and not client_name:
            st.caption("Enter a client name before uploading or resuming. "
                       "An empty name mixes books on this shared instance.")

        if live:
            _render_resume_controls(client_name)

        st.markdown("### 2 · Transactions")
        upload_help = (
            "The file is stored on this server's persistent volume for this "
            "client. Do not upload data you are not authorized to process."
            if live else
            "The file stays on this computer — nothing is sent online.")
        uploaded = st.file_uploader(
            "Choose a file (.xlsx or .csv)",
            type=["xlsx", "xlsm", "xls", "csv"], accept_multiple_files=False,
            help=upload_help)

        sheet = 0
        if uploaded is not None:
            raw = uploaded.getvalue()
            is_csv = uploaded.name.lower().endswith(".csv")
            sheets = list_excel_sheets(raw) if not is_csv else []
            if sheets:
                sheet = st.selectbox("Which tab/sheet?", options=sheets, index=0)
            load_disabled = live and not client_name
            if st.button("Load & open spreadsheet", type="primary",
                         width="stretch", disabled=load_disabled):
                try:
                    with st.spinner("Reading and analysing the file…"):
                        common.load_into_session(
                            raw, uploaded.name,
                            Path(uploaded.name).suffix.lower(),
                            sheet=(sheet if sheets else 0))
                    work = st.session_state["work_df"]
                    loaded = st.session_state["loaded"]
                    from src.ingest_profiles import profile_chip
                    common.set_flash(
                        f"Loaded '{uploaded.name}' ({len(work):,} rows) — "
                        f"{profile_chip(loaded.source_profile)}.")
                    st.session_state["view"] = "spreadsheet"
                    st.session_state["panel"] = None
                    st.rerun()
                except DataLoadError as exc:
                    st.error(f"Could not read that file: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected problem while loading: {exc}")
                    logger.error("upload_failed", error=str(exc))

    with right:
        st.markdown("### No file yet?")
        st.write("Load a representative sample dataset to evaluate the workflow.")
        sample_disabled = live and not client_name
        st.button("Load sample dataset", width="stretch",
                  key="landing_sample", on_click=common.load_sample,
                  kwargs={"navigate": True}, disabled=sample_disabled)

        st.markdown("---")
        runs = storage.list_runs(limit=3)
        if runs:
            st.markdown("##### Recent runs")
            for r in runs:
                st.caption(f"{r.file_name or '—'} — {r.auto_filled} auto-filled, "
                           f"{r.needs_review} to review")

        if common.demo_mode():
            st.markdown("##### Demo controls")
            st.button(
                "Reset demo data", width="stretch", key="demo_reset",
                on_click=common.reset_demo_data,
                help="Clear all clients, rules, learned memory, models and run "
                     "history — returns the demo to a brand-new state.")

    if st.session_state.get("work_df") is not None:
        st.divider()
        work = st.session_state["work_df"]
        loaded = st.session_state["loaded"]
        st.success(f"**{loaded.source_name}** is loaded ({len(work):,} rows).")
        st.button("Continue to the spreadsheet", type="primary",
                  key="landing_continue", on_click=common.goto_view,
                  args=("spreadsheet", None))

    common.render_optional_setup()
