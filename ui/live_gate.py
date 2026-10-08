"""Shared-password gate for the hosted client instance.

Imported only after the dependency gate and ``st.set_page_config``. Must run
*before* ``common.services()`` / ``bootstrap()`` so a public visitor cannot
trigger the cached bootstrap (or a demo reset).
"""

from __future__ import annotations

import hmac

import streamlit as st

UNLOCK_KEY = "_live_unlocked"


def enforce_live_auth() -> None:
    """Block the app until the shared password is entered (live only).

    Live + empty ``FILLDOWN_AUTH_PASSWORD`` refuses to render. Local and
    public-demo runs return immediately.
    """
    from src.config import auth_password, is_live, live_boot_blocked

    if not is_live():
        return
    if live_boot_blocked():
        st.error(
            "This hosted client instance will not start until "
            "`FILLDOWN_AUTH_PASSWORD` is set on the service."
        )
        st.caption("Set a long random password on the Railway service, then "
                   "redeploy. The instance stays closed until then.")
        st.stop()

    if st.session_state.get(UNLOCK_KEY):
        return

    st.title("Code_Down_ML")
    st.caption("Hosted client instance")
    st.write("Enter the shared password to continue.")
    pwd = st.text_input("Password", type="password", key="live_auth_password")
    if st.button("Unlock", type="primary", key="live_auth_unlock"):
        expected = auth_password()
        ok = False
        if isinstance(pwd, str) and isinstance(expected, str) and pwd:
            try:
                ok = hmac.compare_digest(pwd, expected)
            except Exception:  # noqa: BLE001
                ok = False
        if ok:
            st.session_state[UNLOCK_KEY] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()
