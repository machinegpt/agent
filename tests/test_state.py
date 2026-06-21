"""Unit tests for jinx.state persistence and merge helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from jinx.state import (
    StateBlock,
    StateManager,
    _resolve_jinx_path,
    merge_state,
    read_jinx,
    write_jinx,
)


class TestResolveJinxPath:
    """Tests for JINX.yaml path resolution."""

    def test_env_override_takes_precedence(self, monkeypatch, tmp_path: Path) -> None:
        env_path = tmp_path / "custom" / "JINX.yaml"
        monkeypatch.setenv("JINX_PATH", str(env_path))
        assert _resolve_jinx_path() == env_path

    def test_env_override_resolves_relative_path(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("JINX_PATH", "custom/JINX.yaml")
        assert _resolve_jinx_path() == (tmp_path / "custom" / "JINX.yaml").resolve()


class TestLoadState:
    """Tests for StateManager.load_state."""

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


class TestPersistState:
    """Tests for StateManager.persist_state atomic writes."""

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
    """Tests for merge_state incremental updates."""

    def test_valid_update_merged_into_state(self) -> None:
        jinx: dict[str, Any] = {"state": {"task": "old"}}
        update = {"task": "new", "facts": ["fact1"], "exit_ready": True}

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "new"
        assert result["state"]["facts"] == ["fact1"]
        assert result["state"]["exit_ready"] is True

    def test_invalid_update_rejected_without_corrupting_state(self) -> None:
        jinx: dict[str, Any] = {"state": {"task": "preserved", "facts": ["keep"]}}
        update = {"task": 12345}  # int violates StateBlock.task Optional[str]

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
        # First entries should have prior_failure stripped
        assert "prior_failure" not in kept[0]
        # The most recent 5 should retain prior_failure
        assert kept[-1].get("prior_failure") == "failure-6"
        assert kept[-3].get("prior_failure") == "failure-4"


class TestLegacyAliases:
    """Tests for read_jinx / write_jinx compatibility helpers."""

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
