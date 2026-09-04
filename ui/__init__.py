"""Streamlit UI layer for ProfitCoach — Fill Down Automation.

The UI is intentionally split from the business logic in ``src/``:

* ``ui.setup``      – dependency status + in-app installer (safe to import even
  before the core packages exist; it only touches ``streamlit`` and stdlib).
* ``ui.common``     – cached bootstrap, session state, flash + shared helpers.
* ``ui.sidebar``    – the minimal sidebar (status, settings, navigation).
* ``ui.landing``    – the Landing view (client + upload).
* ``ui.spreadsheet``– the dominant spreadsheet workspace.
* ``ui.panels``     – secondary views (Rules, Models, History).
"""
