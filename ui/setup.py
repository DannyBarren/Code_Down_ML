"""Dependency status + in-app installer.

This module is deliberately lightweight: it imports only ``streamlit`` and
``src.dependencies`` (standard library only) at module load, so ``main.py`` can
import it *before* verifying that the heavy core packages (pandas, sklearn, …)
are present. Anything that needs those packages is imported lazily inside the
functions that use it.
"""

from __future__ import annotations

import streamlit as st

from src import dependencies as deps


# --------------------------------------------------------------------------- #
# Real (import-tested) availability of the optional engines
# --------------------------------------------------------------------------- #
def semantic_engine_ready() -> tuple[bool, bool]:
    """Return (semantic_ok, setfit_ok) using the engine's own import probes."""
    try:
        from src.ml_classifier import setfit_available
        from src.similarity import semantic_backend_available
        return semantic_backend_available()[0], setfit_available()[0]
    except Exception:  # noqa: BLE001 - before core deps this can fail
        return False, False


# --------------------------------------------------------------------------- #
# Installer
# --------------------------------------------------------------------------- #
def _run_install(kind: str) -> None:
    """Run pip in a subprocess and stream the output into the UI."""
    st.session_state["installing"] = True
    title = ("core engine" if kind == "core"
             else "semantic + ML engine (large download)")
    st.info(f"Installing the {title}. Please keep this window open.")
    log_box = st.empty()
    lines: list[str] = []
    exit_code = 1
    stream = deps.install_core() if kind == "core" else deps.install_semantic()
    for line in stream:
        code = deps.parse_exit_code(line)
        if code is not None:
            exit_code = code
            break
        lines.append(line)
        log_box.code("\n".join(lines[-250:]), language="bash")
    st.session_state["installing"] = False
    st.session_state["install_result"] = {
        "kind": kind, "exit_code": exit_code, "log": lines[-250:],
    }
    st.rerun()


def _render_install_result() -> None:
    res = st.session_state.get("install_result")
    if not res:
        return
    label = "Core engine" if res["kind"] == "core" else "Semantic + ML engine"
    if res["exit_code"] == 0:
        st.success(f"{label} installed successfully. Select **Restart App** "
                   "below to enable the new features.")
        st.caption("If the features don't appear after restarting, fully close "
                   "this window and re-launch with: streamlit run main.py")
    else:
        st.error(f"{label} installation did not complete "
                 f"(exit code {res['exit_code']}). Review the log below and "
                 "retry, or run the command shown manually.")
        with st.expander("Installation log"):
            st.code("\n".join(res["log"]), language="bash")
    c1, c2 = st.columns(2)
    if c1.button("Restart App", type="primary", width="stretch",
                 key="restart_after_install"):
        _soft_restart()
    if c2.button("Dismiss", width="stretch", key="dismiss_install_result"):
        st.session_state.pop("install_result", None)
        st.rerun()


def _soft_restart() -> None:
    """Clear caches + dependency probes and rerun."""
    st.session_state.pop("install_result", None)
    st.session_state.pop("show_install_page", None)
    st.cache_resource.clear()
    st.cache_data.clear()
    try:
        from src.ml_classifier import setfit_available
        from src.similarity import semantic_backend_available
        for fn in (semantic_backend_available, setfit_available):
            cache_clear = getattr(fn, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
    except Exception:  # noqa: BLE001
        pass
    st.rerun()


def render_install_buttons(*, include_core: bool, include_semantic: bool,
                           key_prefix: str) -> None:
    if include_core:
        if st.button("Install Core (required, ~1 minute)", type="primary",
                     width="stretch", key=f"{key_prefix}_core",
                     help="Installs the small core packages needed to run. "
                          "Only missing packages are installed."):
            _run_install("core")
    if include_semantic:
        if st.button("Install Semantic + ML (optional, several minutes)",
                     type="primary" if not include_core else "secondary",
                     width="stretch", key=f"{key_prefix}_semantic",
                     help="Large download (~hundreds of MB): torch, "
                          "sentence-transformers and SetFit. Optional — the app "
                          "works without it. Only missing packages are installed."):
            _run_install("semantic")


def render_setup_screen(status: deps.DependencyStatus) -> None:
    """'Setup Required' screen shown when core deps are missing."""
    st.title("Code_Down_ML — Setup Required")
    st.write(
        "A few core packages need to be installed on this computer before the "
        "application can run. Select the button below and the installer runs "
        "automatically. This is a one-time step.")

    st.markdown("### Current status")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_status_line(
            f"Core engine ({len(status.core.present)}/{status.core.total})",
            status.core_ok))
        st.caption(_ENGINE_BLURB["core"])
    with c2:
        st.markdown(_status_line("Semantic AI", status.semantic_ok, optional=True))
        st.caption(_ENGINE_BLURB["semantic"])
    st.caption("Missing core packages: " + ", ".join(status.core_missing))

    st.divider()
    st.markdown("### One-click setup")
    st.write(
        "Select **Install Core** to enable the application (about one minute). "
        "The optional semantic + SetFit engines can be added later from the "
        "sidebar and are not required to start.")

    _render_install_result()
    if not st.session_state.get("install_result"):
        render_install_buttons(include_core=True, include_semantic=True,
                               key_prefix="setup")

    with st.expander("Prefer to install manually?", expanded=False):
        st.write("Run this in a terminal from the app folder, then reopen the app:")
        st.code("python -m pip install -r requirements.txt", language="bash")
        st.write("For the optional advanced engines:")
        st.code("python -m pip install -r requirements-semantic.txt",
                language="bash")


def render_install_page() -> None:
    """Full-page installer reached from the sidebar (core already present)."""
    st.title("Install Dependencies")
    st.write(
        "Install or repair the application's engines here. Only missing packages "
        "are installed — nothing already working is changed or upgraded.")

    status = deps.check_dependencies()
    sem_ok, sf_ok = semantic_engine_ready()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_status_line("Core engine", status.core_ok))
        st.caption(_ENGINE_BLURB["core"])
    with c2:
        st.markdown(_status_line("Semantic AI", sem_ok, optional=True))
        st.caption(_ENGINE_BLURB["semantic"])
    with c3:
        st.markdown(_status_line("SetFit ML", sf_ok, optional=True))
        st.caption(_ENGINE_BLURB["setfit"])

    st.divider()
    _render_install_result()
    if not st.session_state.get("install_result"):
        if sem_ok and sf_ok:
            st.success("All engines are installed.")
        else:
            st.caption("The semantic + ML download is large (hundreds of MB) and "
                       "may take several minutes. Only the missing packages are "
                       "installed — nothing already working is touched.")
            render_install_buttons(include_core=not status.core_ok,
                                   include_semantic=True, key_prefix="page")

    st.divider()
    if st.button("Back to the app", key="install_page_back"):
        st.session_state["show_install_page"] = False
        st.session_state.pop("install_result", None)
        st.rerun()


def _status_line(label: str, ok: bool, optional: bool = False) -> str:
    """A clear, theme-aware status badge (colored text, not emoji)."""
    if ok:
        return f"**{label}** — :green[Ready]"
    if optional:
        return f"**{label}** — :gray[Not installed (optional)]"
    return f"**{label}** — :red[Not installed]"


# Plain-English description of what each engine adds, shown in the installers.
_ENGINE_BLURB = {
    "core": "Runs the app: reliable TF-IDF similarity + a LogReg model. Required.",
    "semantic": "Matches on meaning, not just spelling (e.g. *CloudHost* ~ "
                "*Cloud Hosting*). Optional.",
    "setfit": "Most accurate trainable model; learns from your approvals. "
              "Optional.",
}


def render_sidebar_dependencies(status: deps.DependencyStatus) -> None:
    """Compact three-way (Core / Semantic / SetFit) status + installer entry."""
    sem_ok, sf_ok = semantic_engine_ready()
    all_ok = status.core_ok and sem_ok and sf_ok
    # One-line summary so the headline state is obvious before expanding.
    if all_ok:
        summary = "Engine status — :green[all ready]"
    elif status.core_ok:
        summary = "Engine status — :green[ready to run] · :gray[optional AI available]"
    else:
        summary = "Engine status — :red[setup needed]"
    with st.expander(summary, expanded=not all_ok):
        st.markdown(_status_line("Core engine", status.core_ok))
        st.markdown(_status_line("Semantic AI", sem_ok, optional=True))
        st.markdown(_status_line("SetFit (advanced ML)", sf_ok, optional=True))
        if all_ok:
            st.caption("All matching engines are available.")
        elif sem_ok:
            st.caption("Semantic matching is active. SetFit is not installed; the "
                       "built-in LogReg model is used instead.")
        else:
            st.caption(
                "Optional AI engines are not installed. The app runs on the "
                "reliable TF-IDF + LogReg fill down without them.")
        if not (sem_ok and sf_ok):
            if st.button("Install missing AI engines", width="stretch",
                         key="sidebar_install_missing", type="primary",
                         help="Large download (~hundreds of MB, several minutes). "
                              "Optional — only missing packages are installed."):
                _run_install("semantic")
        if st.button("Install / manage dependencies", width="stretch",
                     key="sidebar_open_install"):
            st.session_state["show_install_page"] = True
            st.rerun()
