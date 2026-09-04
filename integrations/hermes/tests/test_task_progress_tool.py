from __future__ import annotations

import json

from mnemosyne_hermes import MnemosyneMemoryProvider


def _provider(tmp_path, profile: str = "profile_a") -> MnemosyneMemoryProvider:
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        agent_context="primary",
        agent_identity=profile,
    )
    assert provider._beam is not None
    return provider


def _close(provider: MnemosyneMemoryProvider) -> None:
    try:
        provider._beam.conn.close()
    except Exception:
        pass


def test_task_progress_roundtrip_and_list(tmp_path):
    provider = _provider(tmp_path, profile="profile_a")
    try:
        set_result = json.loads(provider.handle_tool_call(
            "mnemosyne_task_progress",
            {
                "action": "set",
                "task": "mnemosyne-pr",
                "state": "Implemented recall diagnostics. Next: open PR.",
                "metadata": {"status": "in_progress"},
            },
        ))
        assert set_result["status"] == "set"
        assert set_result["owner_id"] == "profile_a"

        get_result = json.loads(provider.handle_tool_call(
            "mnemosyne_task_progress",
            {"action": "get", "task": "mnemosyne-pr"},
        ))
        assert get_result["status"] == "found"
        assert get_result["owner_id"] == "profile_a"
        assert "Implemented recall diagnostics" in get_result["state"]
        assert "status" in get_result["state"]

        list_result = json.loads(provider.handle_tool_call(
            "mnemosyne_task_progress",
            {"action": "list"},
        ))
        assert list_result["count"] == 1
        assert list_result["tasks"][0]["task"] == "mnemosyne-pr"
    finally:
        _close(provider)


def test_task_progress_is_profile_scoped(tmp_path):
    provider = _provider(tmp_path, profile="profile_a")
    try:
        provider.handle_tool_call(
            "mnemosyne_task_progress",
            {"action": "set", "task": "shared-name", "state": "Profile A state"},
        )
        provider._agent_identity = "profile_b"
        get_result = json.loads(provider.handle_tool_call(
            "mnemosyne_task_progress",
            {"action": "get", "task": "shared-name"},
        ))
        assert get_result == {"status": "not_found", "task": "shared-name"}
    finally:
        _close(provider)


def test_task_progress_clear_and_validation(tmp_path):
    provider = _provider(tmp_path, profile="profile_a")
    try:
        assert json.loads(provider.handle_tool_call(
            "mnemosyne_task_progress", {"action": "set", "task": "x"}
        )) == {"error": "state is required for set"}

        provider.handle_tool_call(
            "mnemosyne_task_progress",
            {"action": "set", "task": "to-clear", "state": "temporary"},
        )
        cleared = json.loads(provider.handle_tool_call(
            "mnemosyne_task_progress", {"action": "clear", "task": "to-clear"}
        ))
        assert cleared == {"status": "cleared", "task": "to-clear"}

        missing = json.loads(provider.handle_tool_call(
            "mnemosyne_task_progress", {"action": "get", "task": "to-clear"}
        ))
        assert missing == {"status": "not_found", "task": "to-clear"}
    finally:
        _close(provider)


def test_task_progress_schema_exposed():
    provider = MnemosyneMemoryProvider()
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert "mnemosyne_task_progress" in names
