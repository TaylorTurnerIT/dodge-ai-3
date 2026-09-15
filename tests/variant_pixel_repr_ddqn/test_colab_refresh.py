"""Proxy refresh must preserve the existing runtime and kernel."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "colab_refresh",
    Path(__file__).parents[2] / "variants/pixel-repr-ddqn/scripts/colab_refresh.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_refresh_preserves_runtime_and_kernel():
    session = SimpleNamespace(
        endpoint="expected",
        token="old",
        url="old",
        kernel_id="kernel",
        session_id="session",
        keep_alive_pid=42,
    )
    proxy = SimpleNamespace(token="new", url="new", token_expires_in_seconds=3600)
    saved = []
    state = SimpleNamespace(
        client=SimpleNamespace(
            list_assignments=lambda: [
                SimpleNamespace(endpoint="expected", runtime_proxy_info=proxy)
            ]
        ),
        store=SimpleNamespace(get=lambda _: session, add=saved.append),
    )
    assert module.refresh(state, "name", "expected") == 3600
    assert saved == [session]
    assert (session.token, session.url) == ("new", "new")
    assert (session.kernel_id, session.session_id, session.keep_alive_pid) == (
        "kernel",
        "session",
        42,
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        module.refresh(state, "name", "other")
    session.endpoint = "other"
    with pytest.raises(RuntimeError, match="does not match"):
        module.refresh(state, "name", "expected")
