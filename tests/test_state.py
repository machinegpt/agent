"""State management and persistence regression tests for JINX.

These tests cover the contract of the state manifest, validation logic, and
stale-run detection without relying on real waiting or sleeping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from jinx.runner import Yaml
from jinx.state import (
    StateManager,
    _resolve_jinx_path,
    merge_state,
    read_jinx,
    write_jinx,
)


class TestResolveJinxPath:
    """Path resolution contract for JINX.yaml."""

    def test_env_override_takes_precedence(self, monkeypatch, tmp_path: Path) -> None:
        env_path = tmp_path / "custom" / "JINX.yaml"
        monkeypatch.setenv("JINX_PATH", str(env_path))
        assert _resolve_jinx_path() == env_path

    def test_env_override_resolves_relative_path(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("JINX_PATH", "custom/JINX.yaml")
        assert _resolve_jinx_path() == (tmp_path / "custom" / "JINX.yaml").resolve()


class TestStateManagerLoad:
    """Loading behaviour for the on-disk state manifest."""

    def test_missing_file_returns_empty_dict(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("JINX_PATH", str(tmp_path / "nonexistent.yaml"))
        assert StateManager.load_state() == {}

    def test_valid_yaml_loaded(self, monkeypatch, tmp_path: Path) -> None:
        jinx_path = tmp_path / "JINX.yaml"
        expected = {"state": {"task": "test task"}, "meta": {"version": 1}}
        jinx_path.write_text(yaml.safe_dump(expected), encoding="utf-8")
        monkeypatch.setenv("JINX_PATH", str(jinx_path))
        assert StateManager.load_state() == expected

    def test_non_dict_yaml_returns_empty_dict(self, monkeypatch, tmp_path: Path) -> None:
        jinx_path = tmp_path / "JINX.yaml"
        jinx_path.write_text("- just\n- a list\n", encoding="utf-8")
        monkeypatch.setenv("JINX_PATH", str(jinx_path))
        assert StateManager.load_state() == {}

    def test_invalid_yaml_returns_empty_dict(self, monkeypatch, tmp_path: Path) -> None:
        jinx_path = tmp_path / "JINX.yaml"
        jinx_path.write_text("not: valid: yaml: [", encoding="utf-8")
        monkeypatch.setenv("JINX_PATH", str(jinx_path))
        assert StateManager.load_state() == {}


class TestStateManagerPersist:
    """Atomic write and persist semantics for the state file."""

    def test_writes_yaml_round_trip(self, monkeypatch, tmp_path: Path) -> None:
        jinx_path = tmp_path / "JINX.yaml"
        monkeypatch.setenv("JINX_PATH", str(jinx_path))
        data = {"state": {"task": "persist me"}, "meta": {"version": 2}}

        StateManager.persist_state(data)

        assert jinx_path.exists()
        assert StateManager.load_state() == data

    def test_creates_parent_directory(self, monkeypatch, tmp_path: Path) -> None:
        jinx_path = tmp_path / "nested" / "deep" / "JINX.yaml"
        monkeypatch.setenv("JINX_PATH", str(jinx_path))

        StateManager.persist_state({"state": {}})

        assert jinx_path.parent.is_dir()
        assert jinx_path.exists()

    def test_overwrites_existing_file(self, monkeypatch, tmp_path: Path) -> None:
        jinx_path = tmp_path / "JINX.yaml"
        jinx_path.write_text("state:\n  task: old\n", encoding="utf-8")
        monkeypatch.setenv("JINX_PATH", str(jinx_path))

        StateManager.persist_state({"state": {"task": "new"}})

        loaded = StateManager.load_state()
        assert loaded["state"]["task"] == "new"


class TestMergeState:
    """Incremental merge validation and field preservation."""

    def test_valid_update_merged_into_state(self) -> None:
        jinx: dict[str, Any] = {"state": {"task": "old"}}
        update = {"task": "new", "facts": ["fact1"], "exit_ready": True}

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "new"
        assert result["state"]["facts"] == ["fact1"]
        assert result["state"]["exit_ready"] is True

    def test_invalid_update_rejected_without_corrupting_state(self) -> None:
        jinx: dict[str, Any] = {"state": {"task": "preserved", "facts": ["keep"]}}
        update = {"task": 12345}

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "preserved"
        assert result["state"]["facts"] == ["keep"]

    def test_none_values_do_not_overwrite_existing_state(self) -> None:
        jinx: dict[str, Any] = {"state": {"task": "existing", "facts": ["a"]}}
        update: dict[str, Any] = {"exit_ready": True}

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "existing"
        assert result["state"]["facts"] == ["a"]
        assert result["state"]["exit_ready"] is True

    def test_none_fields_are_ignored_across_all_state_keys(self) -> None:
        jinx: dict[str, Any] = {
            "state": {
                "task": "existing-task",
                "facts": ["keep"],
                "debt": ["debt-old"],
                "open": ["issue-old"],
                "scores": [{"round": 1, "all_pass": False, "pass_count": 1}],
            }
        }
        update: dict[str, Any] = {
            "task": None,
            "facts": None,
            "debt": None,
            "open": None,
            "scores": None,
            "exit_ready": False,
            "deadlock": True,
        }

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "existing-task"
        assert result["state"]["facts"] == ["keep"]
        assert result["state"]["debt"] == ["debt-old"]
        assert result["state"]["open"] == ["issue-old"]
        assert result["state"]["scores"][0]["round"] == 1
        assert result["state"]["deadlock"] is True

    def test_simplified_score_verdict_is_normalized(self) -> None:
        jinx: dict[str, Any] = {"state": {}}
        update = {"scores": [{"round": 1, "verdict": "pass", "detail": "works now"}]}

        result = merge_state(jinx, update)

        assert result["state"]["scores"][0]["all_pass"] is True
        assert result["state"]["scores"][0]["pass_count"] == 1
        assert result["state"]["scores"][0]["requirements"] == {"task_complete": True}

    def test_simplified_score_non_string_detail_is_coerced_without_error(self) -> None:
        jinx: dict[str, Any] = {"state": {}}
        for detail in (["a", "b"], {"reason": "bad"}, 17):
            update = {"scores": [{"round": 1, "verdict": "pass", "detail": detail}]}

            result = merge_state(jinx, update)

            entry = result["state"]["scores"][0]
            assert entry["all_pass"] is True
            assert entry["requirements"] == {"task_complete": True}
            assert isinstance(entry["approach"], str)
            assert len(entry["approach"]) <= 80

    def test_simplified_score_falsy_detail_values_are_preserved_as_text(self) -> None:
        jinx: dict[str, Any] = {"state": {}}
        for detail in (0, False):
            update = {"scores": [{"round": 1, "verdict": "pass", "detail": detail}]}

            result = merge_state(jinx, update)

            entry = result["state"]["scores"][0]
            assert entry["approach"] == str(detail)
            assert entry["all_pass"] is True

    def test_yaml_dump_preserves_multiline_field_values(self) -> None:
        payload = {
            "task": "first line\n\nthird line",
            "facts": ["alpha", "beta\n\ngamma"],
        }

        dumped = Yaml.dump_to_string(payload)
        restored = yaml.safe_load(dumped)

        assert restored == payload

    def test_score_history_truncates_prior_failure(self) -> None:
        jinx: dict[str, Any] = {"state": {}}
        scores = [
            {
                "round": i,
                "approach": f"approach-{i}",
                "prior_failure": f"failure-{i}",
                "requirements": {},
            }
            for i in range(7)
        ]
        update = {"scores": scores, "exit_ready": False}

        result = merge_state(jinx, update)

        kept = result["state"]["scores"]
        assert len(kept) == 7
        assert "prior_failure" not in kept[0]
        assert kept[-1].get("prior_failure") == "failure-6"
        assert kept[-3].get("prior_failure") == "failure-4"


class TestHistoryCompaction:
    """History compaction logic for request payloads."""

    def test_compact_history_keeps_recent_messages_only(self) -> None:
        from jinx.runner import compact_history_for_request

        history = [
            {"role": "user", "content": "ROUND 1/10 — TASK: first\nSTATE:\nstate-1"},
            {"role": "assistant", "content": [{"type": "text", "text": "reply-1"}]},
            {"role": "user", "content": "ROUND 2/10 — TASK: first\nSTATE:\nstate-2"},
            {"role": "assistant", "content": [{"type": "text", "text": "reply-2"}]},
            {"role": "user", "content": "ROUND 3/10 — TASK: first\nSTATE:\nstate-3"},
            {"role": "assistant", "content": [{"type": "text", "text": "reply-3"}]},
            {"role": "user", "content": "ROUND 4/10 — TASK: first\nSTATE:\nstate-4"},
            {"role": "assistant", "content": [{"type": "text", "text": "reply-4"}]},
        ]

        compact = compact_history_for_request(history, max_messages=4)

        assert len(compact) == 4
        assert all("ROUND 1/10" not in str(msg["content"]) for msg in compact)
        assert compact[-1]["content"] == [{"type": "text", "text": "reply-4"}]


class TestStaleRunDetection:
    """Guard against false stale-session detection without real waits."""

    def test_recent_file_activity_prevents_stale_timeout(self, monkeypatch, tmp_path: Path) -> None:
        from jinx.runner import _is_run_state_stale

        request_path = tmp_path / "jinx_request.yaml"
        run_state_path = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request_path)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state_path)

        request_path.write_text("type: llm_generate\n", encoding="utf-8")
        run_state_path.write_text("waiting_for: llm_generate\n", encoding="utf-8")

        assert _is_run_state_stale({"updated_at": 0}, timeout_seconds=30) is False

    def test_recent_run_state_timestamp_does_not_mark_session_stale(self, monkeypatch) -> None:
        from jinx.runner import _is_run_state_stale

        fake_now = 1_000.0
        monkeypatch.setattr("jinx.runner.time.time", lambda: fake_now)

        run_state = {"updated_at": fake_now - 5}
        assert _is_run_state_stale(run_state, timeout_seconds=30) is False

    def test_stale_session_without_recent_activity_is_detected(self, monkeypatch, tmp_path: Path) -> None:
        from jinx.runner import _is_run_state_stale

        request_path = tmp_path / "jinx_request.yaml"
        run_state_path = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request_path)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state_path)

        fake_now = 2_000.0
        monkeypatch.setattr("jinx.runner.time.time", lambda: fake_now)

        request_path.write_text("type: llm_generate\n", encoding="utf-8")
        run_state_path.write_text("waiting_for: llm_generate\n", encoding="utf-8")

        old_time = fake_now - 60
        os.utime(request_path, (old_time, old_time))
        os.utime(run_state_path, (old_time, old_time))

        assert _is_run_state_stale({"updated_at": old_time}, timeout_seconds=30) is True

    def test_recent_file_activity_is_preferred_over_old_run_state_value(self, monkeypatch, tmp_path: Path) -> None:
        from jinx.runner import _is_run_state_stale

        request_path = tmp_path / "jinx_request.yaml"
        run_state_path = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request_path)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state_path)

        fake_now = 5_000.0
        monkeypatch.setattr("jinx.runner.time.time", lambda: fake_now)

        request_path.write_text("type: llm_generate\n", encoding="utf-8")
        run_state_path.write_text("waiting_for: llm_generate\n", encoding="utf-8")
        os.utime(request_path, (fake_now - 5, fake_now - 5))
        os.utime(run_state_path, (fake_now - 10, fake_now - 10))

        assert _is_run_state_stale({"updated_at": 0}, timeout_seconds=30) is False


class TestLegacyAliases:
    """Compatibility adapters for manifest read/write helpers."""

    def test_read_jinx_delegates_to_state_manager(self, monkeypatch, tmp_path: Path) -> None:
        jinx_path = tmp_path / "JINX.yaml"
        expected = {"state": {"task": "legacy"}}
        jinx_path.write_text(yaml.safe_dump(expected), encoding="utf-8")
        monkeypatch.setenv("JINX_PATH", str(jinx_path))
        assert read_jinx() == expected

    def test_write_jinx_delegates_to_state_manager(self, monkeypatch, tmp_path: Path) -> None:
        jinx_path = tmp_path / "JINX.yaml"
        monkeypatch.setenv("JINX_PATH", str(jinx_path))
        write_jinx({"state": {"task": "written"}})
        assert StateManager.load_state()["state"]["task"] == "written"
