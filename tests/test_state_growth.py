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
import yaml

from jinx.runner import (
    _update_tool_result_cache,
    compact_history_for_request,
    summarize_dropped_history,
    write_llm_request,
)
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

    def test_history_ending_with_a_tool_response(self) -> None:
        """A window whose last message is a tool_result must not orphan it."""
        history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                          "content": "output"}]},
        ]

        for limit in (1, 2, 3, 4):
            win = compact_history_for_request(history, limit)
            assert _orphan_result_ids(win) == [], "limit=%d produced an orphan" % limit

    def test_lone_orphan_tool_result_is_dropped(self) -> None:
        """Regression: the window must not unconditionally keep its last message."""
        history = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "gone",
                                          "content": "output"}]},
        ]

        assert compact_history_for_request(history, 6) == []

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


class TestToolResultMemoization:
    """A retried call must be answerable from memory, not just recognised by id."""

    def test_results_are_recorded_by_tool_use_id(self) -> None:
        cache = _update_tool_result_cache(None, [
            {"tool_use_id": "a", "content": "output A"},
            {"tool_use_id": "b", "content": "output B"},
        ])

        assert cache == {"a": "output A", "b": "output B"}

    def test_none_content_is_memoized_as_empty_string(self) -> None:
        cache = _update_tool_result_cache(None, [{"tool_use_id": "a", "content": None}])

        assert cache == {"a": ""}

    def test_non_string_content_is_serialized(self) -> None:
        cache = _update_tool_result_cache(None, [{"tool_use_id": "a", "content": {"k": "v"}}])

        assert "k" in cache["a"]

    def test_entries_without_an_id_are_ignored(self) -> None:
        cache = _update_tool_result_cache(None, [
            {"content": "orphan result"},
            {"tool_use_id": "", "content": "blank id"},
            {"tool_use_id": 7, "content": "numeric id"},
        ])

        assert cache == {}

    def test_cache_is_bounded_and_keeps_newest(self, monkeypatch) -> None:
        monkeypatch.setattr("jinx.runner.TOOL_RESULT_CACHE_CAP", 3)

        cache = _update_tool_result_cache(None, [
            {"tool_use_id": "t%d" % i, "content": str(i)} for i in range(6)
        ])

        assert list(cache) == ["t3", "t4", "t5"]

    def test_re_recording_moves_an_id_to_the_newest_slot(self, monkeypatch) -> None:
        monkeypatch.setattr("jinx.runner.TOOL_RESULT_CACHE_CAP", 3)
        cache = _update_tool_result_cache(
            {"t1": "1", "t2": "2", "t3": "3"}, [{"tool_use_id": "t1", "content": "1b"}]
        )

        assert list(cache) == ["t2", "t3", "t1"], "t1 should be newest, not oldest"
        assert cache["t1"] == "1b"

    def test_existing_cache_survives_an_empty_update(self) -> None:
        cache = _update_tool_result_cache({"a": "kept"}, [])

        assert cache == {"a": "kept"}


class TestRejectedBlockCannotTerminateTheLoop:
    """exit_ready/deadlock must be honoured only from validated state."""

    def test_outcome_reports_acceptance(self) -> None:
        outcome: dict = {}
        merge_state({"state": {}}, _wrap([_score(1)]), outcome=outcome)

        assert outcome.get("applied") is True

    def test_outcome_reports_rejection(self) -> None:
        outcome: dict = {}
        merge_state({"state": {}}, {"state": {"scores": "bad"}}, outcome=outcome)

        assert outcome.get("applied") is False
        assert "error" in outcome

    def test_rejection_diagnostic_mentions_flags_were_ignored(self) -> None:
        diagnostics: list = []
        merge_state({"state": {}}, {"state": {"scores": "bad"}}, diagnostics=diagnostics)

        assert "flags were NOT honoured" in diagnostics[0]

    def test_rejected_block_leaves_prior_scores_intact(self) -> None:
        jinx = {"state": {"scores": [_score(1)], "exit_ready": False}}
        outcome: dict = {}

        merge_state(jinx, {"state": {"scores": "bad", "exit_ready": True}}, outcome=outcome)

        assert outcome["applied"] is False
        assert jinx["state"]["scores"] == [_score(1)]
        assert jinx["state"]["exit_ready"] is False


class TestUnnumberedScoresDoNotDisplaceTheCurrentRound:
    """check_exit reads scores[-1], so legacy entries must not sit last."""

    def test_unnumbered_entries_come_first(self) -> None:
        merged = merge_scores(
            [{"approach": "legacy"}], [_score(1), _score(2)]
        )

        assert [s.get("round") for s in merged] == [None, 1, 2]

    def test_current_round_is_last(self) -> None:
        """check_exit() reads scores[-1]; the current round must be there."""
        merged = merge_scores([{"approach": "legacy"}], [_score(7)])

        assert merged[-1]["round"] == 7
        assert len(merged) == 2


class TestHistoryCompactionNotice:
    """Finding 2: the notice must be wired into the request, not merely defined."""

    def test_notice_is_none_when_nothing_was_dropped(self) -> None:
        history = [{"role": "user", "content": "a"}]

        assert summarize_dropped_history([], history) is None

    def test_notice_counts_dropped_messages(self) -> None:
        dropped = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

        notice = summarize_dropped_history(dropped, [{"role": "user", "content": "c"}])

        assert notice is not None
        assert "2 earlier message" in notice["content"]

    def test_notice_counts_every_message_the_model_did_not_see(
        self, tmp_path, monkeypatch
    ) -> None:
        """The notice describes the sent window, not the larger persisted one.

        The persist window is deliberately wider than the send window, so
        measuring against it hides the messages living in the gap and
        under-reports the elided count on every round.
        """
        request = tmp_path / "jinx_request.yaml"
        run_state = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state)

        history = [{"role": "user", "content": "m%d" % i} for i in range(20)]
        write_llm_request(history, 1, 0, 2)

        payload = yaml.safe_load(request.read_text(encoding="utf-8"))
        sent = [m for m in payload["messages"] if "elided" not in str(m.get("content", ""))]
        persisted = yaml.safe_load(run_state.read_text(encoding="utf-8"))["history"]

        notice = payload["messages"][0]["content"]
        reported = int(notice.split("]")[1].split("earlier")[0].strip())
        assert reported == len(history) - len(sent), (
            "notice must count every message omitted from the request; "
            "persist window is %d but send window is %d"
            % (len(persisted), len(sent))
        )
        assert len(persisted) > len(sent), "the persist window must still be wider"

    def test_write_llm_request_prepends_the_notice(self, tmp_path, monkeypatch) -> None:
        request = tmp_path / "jinx_request.yaml"
        run_state = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state)

        history = [{"role": "user", "content": "m%d" % i} for i in range(20)]
        write_llm_request(history, 3, 0, 2)

        payload = yaml.safe_load(request.read_text(encoding="utf-8"))
        first = payload["messages"][0]
        assert "elided" in first["content"]
        assert payload["messages"][0]["role"] == "user"

        persisted = yaml.safe_load(run_state.read_text(encoding="utf-8"))
        assert not any(
            isinstance(m.get("content"), str) and "elided" in m["content"]
            for m in persisted["history"]
        ), "the synthetic notice must not be persisted, or it would accumulate"

    def test_write_llm_request_carries_the_result_cache(self, tmp_path, monkeypatch) -> None:
        request = tmp_path / "jinx_request.yaml"
        run_state = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state)
        run_state.write_text(
            yaml.safe_dump({"tool_result_cache": {"a": "cached output"}}),
            encoding="utf-8",
        )

        write_llm_request([{"role": "user", "content": "hi"}], 1, 0, 2)

        payload = yaml.safe_load(request.read_text(encoding="utf-8"))
        assert payload["tool_result_cache"] == {"a": "cached output"}
        persisted = yaml.safe_load(run_state.read_text(encoding="utf-8"))
        assert persisted["tool_result_cache"] == {"a": "cached output"}, \
            "the cache must survive a run-state rewrite"
