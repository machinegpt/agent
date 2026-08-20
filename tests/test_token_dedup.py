"""Regression tests for token deduplication and state merge fixes.

Ensures that:
- construct_round_prompt does not emit a redundant TASK: line.
- write_llm_request / _write_tool_request no longer store task in run_state.
- merge_state correctly extracts nested state from full YAML blocks
  (containing id/protocol/state keys) and merges protocol into jinx.
- All function signatures that lost the task parameter have correct call sites.
"""

from __future__ import annotations

import inspect
from typing import Any

import yaml

from jinx.prompts import MISSING_STATE_WARNING, construct_round_prompt
from jinx.runner import (
    _handle_llm_response,
    _handle_tool_response,
    _write_tool_request,
    write_llm_request,
)
from jinx.state import merge_state


class TestConstructRoundPromptNoRedundantTask:
    """construct_round_prompt must not emit a separate TASK: line."""

    def test_no_task_line_in_output(self) -> None:
        state_dump = "task: Build the feature\nfacts: []\nscores: []"
        result = construct_round_prompt(rnd=1, min_rounds=2, state_dump=state_dump)

        assert "TASK:" not in result, (
            "construct_round_prompt must not contain a separate TASK: line. "
            "Task is already inside state_dump."
        )

    def test_task_appears_exactly_once_via_state_dump(self) -> None:
        task = "UniqueTask12345"
        state_dump = f"task: '{task}'\nfacts: []\nscores: []"
        result = construct_round_prompt(rnd=1, min_rounds=2, state_dump=state_dump)

        count = result.count(task)
        assert count == 1, (
            f"Task should appear exactly once in prompt (inside state_dump), "
            f"but found {count} occurrences."
        )

    def test_state_dump_content_is_present(self) -> None:
        state_dump = "task: my task\nfacts:\n- fact1\nscores: []"
        result = construct_round_prompt(rnd=3, min_rounds=5, state_dump=state_dump)

        assert "ROUND 3" in result
        assert "at least 5 rounds" in result
        assert "CURRENT STATE:" in result
        assert "fact1" in result

    def test_missing_state_warning_prefix(self) -> None:
        state_dump = "task: test\nfacts: []"
        result = construct_round_prompt(
            rnd=1, min_rounds=2, state_dump=state_dump, missing_state=True
        )

        assert result.startswith(MISSING_STATE_WARNING)
        assert "CURRENT STATE:" in result

    def test_no_task_parameter_accepted(self) -> None:
        """Ensure the task parameter was removed from the signature."""
        sig = inspect.signature(construct_round_prompt)
        assert "task" not in sig.parameters, (
            "construct_round_prompt should not accept a 'task' parameter. "
            "Task is provided inside state_dump."
        )


class TestWriteFunctionsNoTaskInRunState:
    """write_llm_request and _write_tool_request must not store task in run_state."""

    def test_write_llm_request_signature_has_no_task(self) -> None:
        sig = inspect.signature(write_llm_request)
        assert "task" not in sig.parameters, (
            "write_llm_request should not accept a 'task' parameter. "
            "Task is persisted in JINX.yaml via _init_new_session."
        )

    def test_write_tool_request_signature_has_no_task(self) -> None:
        sig = inspect.signature(_write_tool_request)
        assert "task" not in sig.parameters, (
            "_write_tool_request should not accept a 'task' parameter."
        )

    def test_handle_llm_response_signature_has_no_task(self) -> None:
        sig = inspect.signature(_handle_llm_response)
        assert "task" not in sig.parameters, (
            "_handle_llm_response should not accept a 'task' parameter."
        )

    def test_handle_tool_response_signature_has_no_task(self) -> None:
        sig = inspect.signature(_handle_tool_response)
        assert "task" not in sig.parameters, (
            "_handle_tool_response should not accept a 'task' parameter."
        )


class TestMergeStateNestedBlock:
    """merge_state must extract nested state from full YAML blocks."""

    def test_full_yaml_block_with_id_and_protocol(self) -> None:
        jinx: dict[str, Any] = {"state": {"task": "old"}}
        update = {
            "id": "JINX",
            "protocol": {"loop": {"min": 2}},
            "state": {
                "task": "new task",
                "facts": ["fact1"],
                "scores": [],
                "debt": [],
                "open": [],
                "exit_ready": True,
                "deadlock": False,
            },
        }

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "new task"
        assert result["state"]["facts"] == ["fact1"]
        assert result["state"]["exit_ready"] is True

    def test_flat_state_without_nesting(self) -> None:
        jinx: dict[str, Any] = {"state": {}}
        update = {
            "task": "flat task",
            "facts": [],
            "scores": [],
            "debt": [],
            "open": [],
            "exit_ready": False,
            "deadlock": False,
        }

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "flat task"

    def test_protocol_is_merged_into_jinx_toplevel(self) -> None:
        jinx: dict[str, Any] = {"state": {}}
        update = {
            "id": "JINX",
            "protocol": {"loop": {"min": 5}},
            "state": {
                "task": "test",
                "facts": [],
                "scores": [],
                "debt": [],
                "open": [],
                "exit_ready": False,
                "deadlock": False,
            },
        }

        result = merge_state(jinx, update)

        assert "protocol" in result
        assert result["protocol"]["loop"]["min"] == 5

    def test_protocol_update_merges_with_existing(self) -> None:
        jinx: dict[str, Any] = {
            "state": {},
            "protocol": {"loop": {"min": 3}, "other": "keep"},
        }
        update = {
            "id": "JINX",
            "protocol": {"loop": {"min": 7}},
            "state": {
                "task": "test",
                "facts": [],
                "scores": [],
                "debt": [],
                "open": [],
                "exit_ready": False,
                "deadlock": False,
            },
        }

        result = merge_state(jinx, update)

        assert result["protocol"]["loop"]["min"] == 7
        assert result["protocol"]["other"] == "keep"

    def test_nested_block_preserves_existing_state_fields(self) -> None:
        jinx: dict[str, Any] = {
            "state": {
                "task": "existing",
                "facts": ["keep_me"],
                "scores": [{"round": 1, "all_pass": True, "pass_count": 1}],
            }
        }
        update = {
            "id": "JINX",
            "protocol": {"loop": {"min": 2}},
            "state": {
                "task": "updated",
                "facts": ["new_fact"],
                "scores": [],
                "debt": [],
                "open": [],
                "exit_ready": False,
                "deadlock": False,
            },
        }

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "updated"
        assert result["state"]["facts"] == ["new_fact"]

    def test_nested_block_with_invalid_state_is_rejected(self) -> None:
        jinx: dict[str, Any] = {"state": {"task": "preserved"}}
        update = {
            "id": "JINX",
            "protocol": {"loop": {"min": 2}},
            "state": {
                "task": 12345,
                "facts": [],
                "scores": [],
                "debt": [],
                "open": [],
                "exit_ready": False,
                "deadlock": False,
            },
        }

        result = merge_state(jinx, update)

        assert result["state"]["task"] == "preserved"


class TestParseStateBlockRegex:
    """parse_state_block must correctly extract YAML from markdown code fences."""

    def test_extracts_last_yaml_block(self) -> None:
        from jinx.runner import parse_state_block

        text = (
            "Some analysis text.\n"
            "```yaml\n"
            "id: JINX\n"
            "state:\n"
            "  task: test\n"
            "  facts: []\n"
            "  scores: []\n"
            "  debt: []\n"
            "  open: []\n"
            "  exit_ready: true\n"
            "  deadlock: false\n"
            "```\n"
        )

        result = parse_state_block(text)

        assert result is not None
        assert "state" in result
        assert result["state"]["task"] == "test"
        assert result["state"]["exit_ready"] is True

    def test_returns_none_for_no_code_block(self) -> None:
        from jinx.runner import parse_state_block

        result = parse_state_block("Just plain text with no code blocks.")

        assert result is None

    def test_returns_none_for_invalid_yaml_in_block(self) -> None:
        from jinx.runner import parse_state_block

        text = "```yaml\nnot: valid: yaml: [\n```"

        result = parse_state_block(text)

        assert result is None

    def test_selects_last_valid_block_when_multiple_present(self) -> None:
        from jinx.runner import parse_state_block

        text = (
            "First block:\n"
            "```yaml\n"
            "some: data\n"
            "```\n"
            "Second block:\n"
            "```yaml\n"
            "id: JINX\n"
            "state:\n"
            "  task: final\n"
            "  facts: []\n"
            "  scores: []\n"
            "  debt: []\n"
            "  open: []\n"
            "  exit_ready: true\n"
            "  deadlock: false\n"
            "```\n"
        )

        result = parse_state_block(text)

        assert result is not None
        assert result["state"]["task"] == "final"

    def test_recognizes_plain_triple_backtick_without_language(self) -> None:
        from jinx.runner import parse_state_block

        text = (
            "```\n"
            "id: JINX\n"
            "state:\n"
            "  task: no lang tag\n"
            "  facts: []\n"
            "  scores: []\n"
            "  debt: []\n"
            "  open: []\n"
            "  exit_ready: false\n"
            "  deadlock: false\n"
            "```\n"
        )

        result = parse_state_block(text)

        assert result is not None
        assert result["state"]["task"] == "no lang tag"
