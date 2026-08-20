"""Regression tests for the JINX exit policy.

These tests validate the precise contract of ``check_exit`` and guard against
common regressions in score history handling.
"""

from __future__ import annotations

from typing import Any

from jinx.runner import check_exit
from jinx.state import ScoreEntry


class TestCheckExit:
    """Behavioral coverage for JINX loop termination."""

    def test_no_scores_returns_false(self) -> None:
        assert check_exit([], min_rounds=10, rnd=1) is False

    def test_single_pass_score_before_min_rounds_returns_false(self) -> None:
        scores: list[dict[str, Any]] = [{"round": 1, "all_pass": True, "pass_count": 5}]
        assert check_exit(scores, min_rounds=10, rnd=1) is False

    def test_single_non_pass_score_returns_false(self) -> None:
        scores: list[dict[str, Any]] = [{"round": 1, "all_pass": False, "pass_count": 2}]
        assert check_exit(scores, min_rounds=10, rnd=1) is False

    def test_two_all_pass_scores_can_exit(self) -> None:
        scores: list[dict[str, Any]] = [
            {"round": 1, "all_pass": True, "pass_count": 3},
            {"round": 2, "all_pass": True, "pass_count": 3},
        ]
        assert check_exit(scores, min_rounds=1, rnd=2) is True

    def test_three_all_pass_scores_are_considered_stable(self) -> None:
        scores = [{"round": i, "all_pass": True, "pass_count": 4} for i in range(1, 4)]
        assert check_exit(scores, min_rounds=1, rnd=3) is True

    def test_recent_improvement_blocks_exit(self) -> None:
        scores: list[dict[str, Any]] = [
            {"round": 1, "all_pass": False, "pass_count": 1},
            {"round": 2, "all_pass": False, "pass_count": 2},
            {"round": 3, "all_pass": False, "pass_count": 3},
            {"round": 4, "all_pass": True, "pass_count": 5},
        ]
        assert check_exit(scores, min_rounds=1, rnd=4) is False

    def test_stable_recent_score_allows_exit(self) -> None:
        scores: list[dict[str, Any]] = [
            {"round": 1, "all_pass": True, "pass_count": 5},
            {"round": 2, "all_pass": True, "pass_count": 5},
            {"round": 3, "all_pass": True, "pass_count": 4},
            {"round": 4, "all_pass": True, "pass_count": 5},
        ]
        assert check_exit(scores, min_rounds=1, rnd=4) is True

    def test_degraded_recent_history_still_exits(self) -> None:
        scores = [{"round": i, "all_pass": True, "pass_count": 5} for i in range(1, 4)] + [
            {"round": i, "all_pass": True, "pass_count": 3} for i in range(4, 7)
        ]
        assert check_exit(scores, min_rounds=1, rnd=6) is True

    def test_latest_failure_blocks_exit(self) -> None:
        scores: list[dict[str, Any]] = [
            {"round": 1, "all_pass": True, "pass_count": 5},
            {"round": 2, "all_pass": False, "pass_count": 2},
        ]
        assert check_exit(scores, min_rounds=1, rnd=2) is False

    def test_long_history_without_progress_does_not_exit(self) -> None:
        scores = [{"round": i, "all_pass": False, "pass_count": 1} for i in range(1, 20)]
        assert check_exit(scores, min_rounds=10, rnd=19) is False

    def test_scoreentry_objects_are_supported(self) -> None:
        scores = [
            ScoreEntry(round=1, all_pass=True, pass_count=5),
            ScoreEntry(round=2, all_pass=True, pass_count=5),
        ]
        assert check_exit(scores, min_rounds=1, rnd=2) is True

    def test_mixed_dict_and_scoreentry_history_is_supported(self) -> None:
        scores: list[dict[str, Any] | ScoreEntry] = [
            {"round": 1, "all_pass": True, "pass_count": 4},
            ScoreEntry(round=2, all_pass=True, pass_count=4),
            {"round": 3, "all_pass": True, "pass_count": 4},
        ]
        assert check_exit(scores, min_rounds=1, rnd=3) is True

    def test_missing_latest_score_value_blocks_exit(self) -> None:
        scores = [
            {"round": 1, "all_pass": True, "pass_count": 5},
            ScoreEntry(round=2),
        ]
        assert check_exit(scores, min_rounds=1, rnd=2) is False

    def test_new_best_score_in_last_window_blocks_exit(self) -> None:
        scores = [
            {"round": 1, "all_pass": True, "pass_count": 5},
            {"round": 2, "all_pass": True, "pass_count": 5},
            {"round": 3, "all_pass": True, "pass_count": 5},
            {"round": 4, "all_pass": True, "pass_count": 6},
        ]
        assert check_exit(scores, min_rounds=1, rnd=4) is False
