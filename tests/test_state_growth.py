"""Regression tests for bounded state growth and lossless score history.

Motivation (measured, not hypothetical): before these changes the score history
was replaced wholesale, so any reply that omitted a round deleted it from disk,
and every round re-sent the whole history in both directions. Persisted run-state
grew quadratically and a single unquoted ':' in the state block silently discarded
everything the model had just produced.
"""

from __future__ import annotations

from typing import Any

import pytest

from jinx.runner import compact_history_for_request
from jinx.state import FACTS_CAP, merge_scores, merge_state, normalize_text_list


def _score(round_no: int, approach: str = "a", passed: bool = False) -> dict:
    return {
        "round": round_no,
        "approach": approach,
        "prior_failure": "why round %d failed" % round_no,
        "requirements": {"req_x": passed, "req_y": not passed},
        "pass_count": 1 if passed else 0,
        "all_pass": passed,
    }


def _wrap(scores: list) -> dict:
    """Wraps score entries in the nested 'state' block the runner actually sends."""
    return {"state": {"scores": scores}}


class TestScoreMergeIsLossless:
    """A delta-only reply must not destroy earlier rounds."""

    def test_delta_only_replies_accumulate_every_round(self) -> None:
        jinx: dict[str, Any] = {"state": {"scores": []}}
        for n in range(1, 6):
            jinx = merge_state(jinx, _wrap([_score(n)]))

        assert [s["round"] for s in jinx["state"]["scores"]] == [1, 2, 3, 4, 5]

    def test_full_resend_still_works_and_does_not_duplicate(self) -> None:
        """Old behaviour must remain valid: models may still re-send everything."""
        jinx: dict[str, Any] = {"state": {"scores": []}}
        for n in range(1, 4):
            jinx = merge_state(jinx, _wrap([_score(n)]))

        jinx = merge_state(jinx, _wrap([_score(1), _score(2), _score(3), _score(4)]))

        rounds = [s["round"] for s in jinx["state"]["scores"]]
        assert rounds == [1, 2, 3, 4]
        assert len(rounds) == len(set(rounds)), "re-sending must not duplicate rounds"

    def test_resent_round_overwrites_in_place(self) -> None:
        jinx: dict[str, Any] = {"state": {"scores": [_score(1, "original")]}}

        jinx = merge_state(jinx, _wrap([_score(1, "revised")]))

        assert len(jinx["state"]["scores"]) == 1
        assert jinx["state"]["scores"][0]["approach"] == "revised"

    def test_merged_order_is_ascending_by_round(self) -> None:
        jinx: dict[str, Any] = {"state": {"scores": [_score(3)]}}
        jinx = merge_state(jinx, _wrap([_score(1), _score(2)]))

        assert [s["round"] for s in jinx["state"]["scores"]] == [1, 2, 3]

    def test_entry_without_int_round_survives(self) -> None:
        jinx = {"state": {"scores": [{"approach": "legacy, no round"}]}}

        jinx = merge_state(jinx, _wrap([_score(1)]))

        approaches = [s.get("approach") for s in jinx["state"]["scores"]]
        assert "legacy, no round" in approaches

    def test_scores_are_sorted_and_deduplicated(self) -> None:
        merged = merge_scores([_score(2), _score(1)], [_score(2), _score(3)])

        assert [s["round"] for s in merged] == [1, 2, 3]

    def test_prior_failure_is_stripped_from_older_rounds_only(self) -> None:
        jinx: dict[str, Any] = {"state": {"scores": []}}
        for n in range(1, 8):
            jinx = merge_state(jinx, _wrap([_score(n)]))

        entries = jinx["state"]["scores"]
        assert "prior_failure" not in entries[0]
        assert "prior_failure" in entries[-1]


class TestValidationFeedback:
    """A rejected block must be visible to the model, not silently dropped."""

    def test_diagnostics_report_rejection_reason(self) -> None:
        jinx = {"state": {"facts": ["kept"]}}
        diagnostics: list = []

        merge_state(jinx, {"state": {"scores": "not-a-list"}}, diagnostics=diagnostics)

        assert jinx["state"]["facts"] == ["kept"], "state must be untouched"
        assert diagnostics, "rejection must be reported"
        assert "REJECTED" in diagnostics[0]

    def test_no_diagnostics_on_successful_merge(self) -> None:
        diagnostics: list = []

        merge_state({"state": {}}, _wrap([_score(1)]), diagnostics=diagnostics)

        assert diagnostics == []

    def test_merge_note_appears_when_history_is_preserved(self) -> None:
        jinx: dict[str, Any] = {"state": {"scores": [_score(1), _score(2)]}}
        diagnostics: list = []

        merge_state(jinx, _wrap([_score(3)]), diagnostics=diagnostics)

        assert len(jinx["state"]["scores"]) == 3
        assert diagnostics and "merged by round" in diagnostics[0]


class TestFactsAreBoundedAndDeduplicated:
    """Facts stay curated (replaced), but cost stays flat."""

    def test_near_duplicates_collapse(self) -> None:
        out = normalize_text_list([
            "Server runs on 3301",
            "server runs on 3301!",
            "  Server runs on 3301  ",
            "Auth uses a bearer token",
            "auth uses a bearer token",
        ])

        assert out == ["Server runs on 3301", "Auth uses a bearer token"]

    def test_cap_keeps_newest_entries(self) -> None:
        out = normalize_text_list(["old1", "old2", "new1", "new2"], cap=2)

        assert out == ["new1", "new2"]

    def test_facts_are_capped_on_merge(self) -> None:
        jinx: dict[str, Any] = {"state": {}}
        noisy = ["fact number %d" % i for i in range(FACTS_CAP + 25)]

        jinx = merge_state(jinx, {"state": {"facts": noisy}})

        assert len(jinx["state"]["facts"]) == FACTS_CAP
        assert jinx["state"]["facts"][-1] == "fact number %d" % (FACTS_CAP + 24)

    def test_facts_are_replaced_not_accumulated(self) -> None:
        """The model must stay able to retract a wrong fact."""
        jinx: dict[str, Any] = {"state": {"facts": ["wrong belief"]}}

        jinx = merge_state(jinx, {"state": {"facts": ["right belief"]}})

        assert jinx["state"]["facts"] == ["right belief"]


def _tool_exchange() -> list:
    return [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash_exec"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]


def _tool_use_ids(win: list) -> set:
    ids = set()
    for m in win:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    ids.add(b.get("id"))
    return ids


def _orphan_result_ids(win: list) -> list:
    known = _tool_use_ids(win)
    orphans = []
    for m in win:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    if b.get("tool_use_id") not in known:
                        orphans.append(b.get("tool_use_id"))
    return orphans


class TestHistoryWindowIsSafe:
    """The window bounds disk growth without producing invalid message sequences."""

    @pytest.mark.parametrize("limit", [1, 2, 3, 4, 5])
    def test_no_orphan_tool_result_at_any_window_size(self, limit: int) -> None:
        assert _orphan_result_ids(compact_history_for_request(_tool_exchange(), limit)) == []

    @pytest.mark.parametrize("limit", [1, 2, 3, 4, 5])
    def test_no_dangling_tool_use_at_the_end(self, limit: int) -> None:
        win = compact_history_for_request(_tool_exchange(), limit)

        tail = win[-1].get("content") if win and isinstance(win[-1].get("content"), list) else []
        assert not [b for b in tail if isinstance(b, dict) and b.get("type") == "tool_use"]

    def test_short_history_is_returned_unchanged(self) -> None:
        history = _tool_exchange()

        assert compact_history_for_request(history, 10) == history

    def test_window_actually_bounds_growth(self) -> None:
        from jinx.runner import Yaml

        history = [
            {"role": "user", "content": "x" * 800} for _ in range(40)
        ]

        full = len(Yaml.dump_to_string({"history": history}).encode("utf-8"))
        windowed = len(
            Yaml.dump_to_string({"history": compact_history_for_request(history, 8)}).encode("utf-8")
        )

        assert windowed < full
