"""Refresh an existing Colab proxy binding without replacing its runtime."""

from __future__ import annotations

import argparse


def refresh(state, session_name: str, endpoint: str) -> int:
    matches = [a for a in state.client.list_assignments() if a.endpoint == endpoint]
    if len(matches) != 1:
        raise RuntimeError("Expected existing Colab assignment is unavailable")
    session = state.store.get(session_name)
    if session is None or session.endpoint != endpoint:
        raise RuntimeError("Local session does not match expected runtime")
    proxy = matches[0].runtime_proxy_info
    session.token = proxy.token
    session.url = proxy.url
    state.store.add(session)
    return proxy.token_expires_in_seconds


def main() -> None:
    # Execute using the installed colab CLI interpreter, not the training venv.
    from colab_cli.auth import AuthProvider
    from colab_cli.common import state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    state.auth_provider = AuthProvider.ADC
    lifetime = refresh(state, args.session, args.endpoint)
    print(f"Existing runtime proxy refreshed; lifetime {lifetime}s")


if __name__ == "__main__":
    main()
