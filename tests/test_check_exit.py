from __future__ import annotations

from typing import Any

from jinx.runner import check_exit


def test_no_scores_returns_false() -> None:
    assert check_exit([], min_rounds=10, rnd=1) is False


def test_single_score_all_pass_returns_false() -> None:
    scores: list[dict[str, Any]] = [
        {"round": 1, "all_pass": True, "pass_count": 5}
    ]
    assert check_exit(scores, min_rounds=10, rnd=1) is False


def test_single_score_no_all_pass_returns_false() -> None:
    scores: list[dict[str, Any]] = [
        {"round": 1, "all_pass": False, "pass_count": 2}
    ]
    assert check_exit(scores, min_rounds=10, rnd=1) is False


def test_two_scores_both_all_pass_returns_true() -> None:
    scores: list[dict[str, Any]] = [
        {"round": 1, "all_pass": True, "pass_count": 3},
        {"round": 2, "all_pass": True, "pass_count": 3},
    ]
    assert check_exit(scores, min_rounds=10, rnd=2) is True


def test_three_scores_all_pass_returns_true() -> None:
    scores = [{"round": i, "all_pass": True, "pass_count": 4} for i in range(1, 4)]
    assert check_exit(scores, min_rounds=10, rnd=3) is True


def test_four_scores_improving_returns_false() -> None:
    scores: list[dict[str, Any]] = [
        {"round": 1, "all_pass": False, "pass_count": 1},
        {"round": 2, "all_pass": False, "pass_count": 2},
        {"round": 3, "all_pass": False, "pass_count": 3},
        {"round": 4, "all_pass": True, "pass_count": 5},
    ]
    assert check_exit(scores, min_rounds=10, rnd=4) is False


def test_four_scores_stable_returns_true() -> None:
    scores: list[dict[str, Any]] = [
        {"round": 1, "all_pass": True, "pass_count": 5},
        {"round": 2, "all_pass": True, "pass_count": 5},
        {"round": 3, "all_pass": True, "pass_count": 4},
        {"round": 4, "all_pass": True, "pass_count": 5},
    ]
    assert check_exit(scores, min_rounds=10, rnd=4) is True


def test_six_scores_degrading_returns_true() -> None:
    scores = [{"round": i, "all_pass": True, "pass_count": 5} for i in range(1, 4)] + [
        {"round": i, "all_pass": True, "pass_count": 3} for i in range(4, 7)
    ]
    assert check_exit(scores, min_rounds=10, rnd=6) is True


def test_latest_not_all_pass_returns_false() -> None:
    scores: list[dict[str, Any]] = [
        {"round": 1, "all_pass": True, "pass_count": 5},
        {"round": 2, "all_pass": False, "pass_count": 2},
    ]
    assert check_exit(scores, min_rounds=10, rnd=2) is False


def test_many_rounds_no_progress_returns_false() -> None:
    scores = [{"round": i, "all_pass": False, "pass_count": 1} for i in range(1, 20)]
    assert check_exit(scores, min_rounds=10, rnd=19) is False
