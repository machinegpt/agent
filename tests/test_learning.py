# Copyright 2026 JINX Enterprise Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the durable lesson ledger and the gated self-patch subsystem."""

import re
import sys
from pathlib import Path

import pytest
import yaml

from jinx import learning, selfpatch
from jinx.learning import (
    add_lessons,
    load_ledger,
    normalize_lesson_text,
    record_outcome,
    render_lessons,
    save_ledger,
)
from jinx.prompts import construct_round_prompt
from jinx.selfpatch import (
    baseline_changed,
    capture_baseline,
    guard_tool_call,
    is_protected_change,
    restore_baseline,
    within_src_tree,
)


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    """An isolated ledger file."""
    path = tmp_path / "lessons.yaml"
    monkeypatch.setattr(learning, "LESSONS_PATH", path)
    return path


@pytest.fixture()
def sandbox_src(tmp_path, monkeypatch):
    """A fake framework source tree plus its baseline directory.

    The layout mirrors the real one (``.agent/src/jinx``) because the baseline
    reaches outside the source tree: ``_repo_root`` walks three levels up from
    SRC_DIR to find the tests it has to protect.
    """
    src = tmp_path / ".agent" / "src" / "jinx"
    src.mkdir(parents=True)
    (src / "state.py").write_text("def merge_state():\n    return {}\n", encoding="utf-8")
    (src / "tools.py").write_text("def tool_schema():\n    return []\n", encoding="utf-8")
    base = tmp_path / "baseline"
    monkeypatch.setattr(selfpatch, "SRC_DIR", src)
    monkeypatch.setattr(selfpatch, "BASELINE_DIR", base)
    return src, base


# ==============================================================================
# Lesson ledger
# ==============================================================================


class TestLessonLedgerIsDurable:
    """The point of the ledger: knowledge must outlive the run that produced it."""

    def test_lessons_do_not_land_in_the_wiped_state_block(self, tmp_path) -> None:
        from jinx.state import merge_state

        jinx = {"state": {"facts": ["keep me"]}}
        outcome = {}
        merge_state(
            jinx,
            {"state": {"facts": ["keep me"], "lessons": ["measure notices against the sent window"]}},
            outcome=outcome,
        )
        assert outcome["applied"] is True
        assert "lessons" not in jinx["state"], (
            "lessons must not be stored in the state block, because "
            "_init_new_session wipes that dict on the next task"
        )
        assert outcome["lessons"] == ["measure notices against the sent window"]

    def test_lessons_survive_a_new_session(self, ledger) -> None:
        save_ledger({"lessons": add_lessons([], ["never re-send the whole score history"])})
        assert load_ledger()["lessons"], "the ledger must persist to its own file"

    def test_round_trip_through_disk(self, ledger) -> None:
        save_ledger({"lessons": add_lessons([], ["rule one", "rule two"])})
        stored = load_ledger()["lessons"]
        assert [l["text"] for l in stored] == ["rule one", "rule two"]

    def test_a_corrupt_ledger_does_not_raise(self, ledger) -> None:
        ledger.write_text("{{{ not yaml", encoding="utf-8")
        assert load_ledger() == {"lessons": []}

    def test_missing_ledger_is_empty_not_an_error(self, ledger) -> None:
        assert load_ledger() == {"lessons": []}


class TestLessonLedgerStaysBounded:
    """An append-only rule list would just re-create the growth defect."""

    def test_repeating_a_lesson_does_not_grow_the_store(self, ledger) -> None:
        stored = add_lessons([], ["always verify the parse"])
        for _ in range(20):
            stored = add_lessons(stored, ["Always verify the parse!"])
        assert len(stored) == 1
        assert stored[0]["seen"] == 21

    def test_cap_is_enforced(self, ledger) -> None:
        stored = add_lessons([], ["rule %d" % i for i in range(50)], cap=10)
        assert len(stored) == 10

    def test_injection_respects_the_count_limit(self, ledger) -> None:
        stored = add_lessons([], ["rule %d" % i for i in range(50)])
        rendered = render_lessons(stored, limit=5, budget=100000)
        assert len(rendered["applied"]) == 5

    def test_injection_respects_the_character_budget(self, ledger) -> None:
        long_rule = "x" * 500
        stored = add_lessons([], [long_rule + " %d" % i for i in range(20)])
        rendered = render_lessons(stored, limit=100, budget=600)
        assert len(rendered["text"]) <= 600, "the budget is a hard cap, not a hint"
        assert len(rendered["applied"]) < 20, "the budget must actually drop entries"

    def test_a_rule_the_ledger_distrusts_is_withheld(self) -> None:
        """A rule blamed more often than confirmed is kept on disk, not shown."""
        lessons = [
            {"text": "blamed rule", "kind": "antipattern", "uses": 3,
             "confirmed": 0, "failed": 4},
            {"text": "proven rule", "kind": "rule", "uses": 2,
             "confirmed": 2, "failed": 0},
        ]
        out = render_lessons(lessons)
        assert "blamed rule" not in out["text"]
        assert "proven rule" in out["text"]
        assert out["applied"] == ["proven rule"], \
            "a withheld rule must not be credited as if it had been shown"

    def test_the_cap_keeps_the_best_rules_not_the_newest(self) -> None:
        """Overwriting the ledger must not silently drop proven rules."""
        existing = [
            {"text": "proven early rule", "kind": "rule", "uses": 2,
             "confirmed": 3, "failed": 0},
            {"text": "proven middle rule", "kind": "rule", "uses": 1,
             "confirmed": 1, "failed": 0},
        ]
        incoming = [{"text": "brand new rule %d" % i} for i in range(3)]
        kept = [entry["text"] for entry in add_lessons(existing, incoming, cap=3)]
        assert len(kept) == 3
        assert "proven early rule" in kept, \
            "a new rule must not push a proven one out of the ledger"
        assert "proven middle rule" in kept

    def test_the_cap_keeps_the_recorded_order_of_equal_scores(self) -> None:
        existing = [
            {"text": "first tied rule", "kind": "rule", "uses": 1,
             "confirmed": 1, "failed": 0},
            {"text": "second tied rule", "kind": "rule", "uses": 1,
             "confirmed": 1, "failed": 0},
            {"text": "third tied rule", "kind": "rule", "uses": 1,
             "confirmed": 1, "failed": 0},
        ]
        kept = [e["text"] for e in add_lessons(existing, [{"text": "newcomer"}], cap=2)]
        assert kept == ["first tied rule", "second tied rule"], \
            "equal scores must keep the order the ledger recorded them in"

    def test_the_budget_covers_the_header_too(self) -> None:
        # The header is text the model reads, so it has to come out of the same
        # budget. Measuring only the rules let the finished block overshoot.
        stored = add_lessons([], ["y" * 40])
        # The rendered block is exactly header + newline + one line, with no
        # trailing newline, so this is the smallest budget that still fits it.
        exact = len(learning.LEARNED_RULES_HEADER) + 1 + len("- " + "y" * 40)
        assert len(render_lessons(stored, limit=10, budget=exact)["text"]) == exact
        assert len(render_lessons(stored, limit=10, budget=exact - 1)["applied"]) == 0, \
            "one character short and the rule must not be claimed as shown"

    def test_applied_matches_the_text_exactly(self) -> None:
        # The old code trimmed the finished string, which could cut the last
        # rule off the text while its key stayed in `applied` — the ledger would
        # then credit the model for a rule it was never shown.
        long_rule = "z" * 300
        stored = add_lessons([], [long_rule + " %d" % i for i in range(10)])
        rendered = render_lessons(stored, limit=100, budget=700)
        body = rendered["text"].split("\n", 1)[1].splitlines()
        assert len(body) == len(rendered["applied"]), \
            "every applied key must correspond to a line the model can see"
        for key, line in zip(rendered["applied"], body):
            assert key in line

    def test_a_budget_too_small_for_the_header_shows_nothing(self) -> None:
        stored = add_lessons([], ["always validate the parse"])
        rendered = render_lessons(stored, limit=10, budget=20)
        assert rendered == {"text": "", "applied": []}, \
            "an honest empty block beats a truncated one"

    def test_render_is_empty_for_an_empty_ledger(self) -> None:
        assert render_lessons([]) == {"text": "", "applied": []}

    def test_normalization_drops_unusable_entries(self) -> None:
        assert normalize_lesson_text(None) is None
        assert normalize_lesson_text("   ") is None
        assert normalize_lesson_text("  a   b  ") == "a b"


class TestLessonsAreCreditAssigned:
    """The ledger must be able to tell a proven rule from a useless one."""

    def test_a_failing_round_blames_the_rules_it_was_shown(self, ledger) -> None:
        stored = add_lessons([], ["try the reverse approach"])
        rendered = render_lessons(stored)
        record_outcome(stored, rendered["applied"], passed=False)
        assert stored[0]["failed"] == 1
        assert stored[0]["uses"] == 1

    def test_a_passing_round_confirms_them(self, ledger) -> None:
        stored = add_lessons([], ["try the reverse approach"])
        record_outcome(stored, render_lessons(stored)["applied"], passed=True)
        assert stored[0]["confirmed"] == 1

    def test_only_shown_lessons_are_settled(self, ledger) -> None:
        shown = add_lessons([], ["shown rule"])
        hidden = add_lessons(shown, ["hidden rule"])
        record_outcome(hidden, render_lessons(shown, limit=1)["applied"], passed=False)
        by_text = {l["text"]: l for l in hidden}
        assert by_text["shown rule"]["uses"] == 1
        assert by_text["hidden rule"].get("uses", 0) == 0

    def test_proven_rules_outrank_unproven_ones(self, ledger) -> None:
        stored = add_lessons([], ["bad rule", "good rule"])
        record_outcome(stored, [learning._normalize("good rule")], passed=True)
        record_outcome(stored, [learning._normalize("good rule")], passed=True)
        record_outcome(stored, [learning._normalize("bad rule")], passed=False)
        record_outcome(stored, [learning._normalize("bad rule")], passed=False)
        top = render_lessons(stored, limit=1)["applied"][0]
        assert top == learning._normalize("good rule")

    def test_no_keys_means_nothing_to_settle(self, ledger) -> None:
        stored = add_lessons([], ["rule"])
        assert record_outcome(stored, [], passed=False) == stored
        assert stored[0].get("uses", 0) == 0

    def test_a_settled_rule_sinks_out_of_a_capped_injection(self, ledger) -> None:
        stored = add_lessons([], ["rule %d" % i for i in range(10)])
        for _ in range(3):
            record_outcome(stored, [learning._normalize("rule 0")], passed=False)
        shown = render_lessons(stored, limit=1)["applied"]
        assert shown != [learning._normalize("rule 0")]


class TestLessonKinds:
    def test_kind_is_inferred_from_wording(self) -> None:
        assert learning._coerce_kind("antipattern") == "antipattern"
        assert learning._coerce_kind("never do X") == "antipattern"
        assert learning._coerce_kind("how to do X") == "skill"
        assert learning._coerce_kind("nonsense") == "rule"

    def test_mapping_form_with_evidence_is_accepted(self, ledger) -> None:
        stored = add_lessons(
            [], [{"text": "check the parse", "kind": "skill", "evidence": "round 2"}]
        )
        assert stored[0]["kind"] == "skill"
        assert stored[0]["evidence"] == "round 2"

    def test_antipatterns_render_with_their_own_marker(self, ledger) -> None:
        stored = add_lessons([], [{"text": "never trust raw flags", "kind": "antipattern"}])
        assert "!" in render_lessons(stored)["text"]


# ==============================================================================
# Prompt injection
# ==============================================================================


class TestLearnedRulesReachThePrompt:
    def test_the_block_is_injected(self, ledger) -> None:
        save_ledger({"lessons": add_lessons([], ["verify the parse before writing"])})
        rendered = render_lessons(load_ledger()["lessons"])
        prompt = construct_round_prompt(1, 2, "state: {}", lessons_text=rendered["text"])
        assert "LEARNED RULES" in prompt
        assert "verify the parse before writing" in prompt

    def test_the_block_is_absent_when_there_is_nothing_to_say(self, ledger) -> None:
        rendered = render_lessons(load_ledger()["lessons"])
        assert rendered["text"] == ""
        prompt = construct_round_prompt(1, 2, "state: {}", lessons_text=rendered["text"])
        assert "LEARNED RULES" not in prompt

    def test_the_prompt_is_unchanged_without_the_parameter(self) -> None:
        assert "LEARNED RULES" not in construct_round_prompt(1, 2, "state: {}")

    def test_the_system_prompt_documents_both_capabilities(self) -> None:
        from jinx.prompts import SYSTEM_PROMPT

        assert "SELF-IMPROVEMENT" in SYSTEM_PROMPT
        assert "lessons" in SYSTEM_PROMPT
        assert ".agent/src/jinx/*.py" in SYSTEM_PROMPT
        assert "You may NOT redefine the brake logic" in SYSTEM_PROMPT


class TestAppliedLessonsSurviveTheProcessBoundary:
    """write_llm_request rebuilds the run state from an explicit key list.

    Anything a caller stashes in the in-memory dict is therefore dropped on
    persist unless it is threaded through explicitly. When `applied_lessons` was
    dropped that way, credit assignment silently never ran: the whole learning
    loop looked wired up and did nothing.
    """

    def test_the_keys_are_persisted(self, tmp_path, monkeypatch) -> None:
        from jinx.runner import write_llm_request

        request = tmp_path / "jinx_request.yaml"
        run_state = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state)

        write_llm_request(
            [{"role": "user", "content": "hi"}], 2, 0, 2,
            applied_lessons=["a key", "another key"],
        )

        persisted = yaml.safe_load(run_state.read_text(encoding="utf-8"))
        assert persisted["applied_lessons"] == ["a key", "another key"]

    def test_a_retry_still_carries_them(self, tmp_path, monkeypatch) -> None:
        from jinx.runner import write_llm_request

        request = tmp_path / "jinx_request.yaml"
        run_state = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state)
        write_llm_request([{"role": "user", "content": "hi"}], 2, 0, 2,
                          applied_lessons=["carried"])

        # A later call that does not pass them explicitly must inherit, not erase.
        write_llm_request([{"role": "user", "content": "hi"}], 2, 0, 2, retry=True)

        persisted = yaml.safe_load(run_state.read_text(encoding="utf-8"))
        assert persisted["applied_lessons"] == ["carried"], \
            "a retry must not silently forget which lessons were shown"

    def test_the_full_credit_loop_closes_across_processes(self, tmp_path, monkeypatch) -> None:
        from jinx import runner

        request = tmp_path / "jinx_request.yaml"
        run_state = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", run_state)
        monkeypatch.setattr(learning, "LESSONS_PATH", tmp_path / "lessons.yaml")

        saved = add_lessons([], ["a proven rule"])
        save_ledger({"lessons": saved})
        rendered = render_lessons(saved)
        runner.write_llm_request(
            [{"role": "user", "content": "hi"}], 2, 0, 2,
            applied_lessons=rendered["applied"],
        )

        # Simulate the next process: reload the run state and settle the round.
        reloaded = yaml.safe_load(run_state.read_text(encoding="utf-8"))
        ledger = load_ledger()
        record_outcome(ledger["lessons"], reloaded.get("applied_lessons"), passed=True)
        assert ledger["lessons"][0]["confirmed"] == 1


class TestTerminalRunCleanup:
    """A finished run should not leave a self-patch baseline behind.

    A stale baseline dir is not harmless: the next task recaptures it anyway, but
    a leftover one silently outlives the run it belonged to, and the sandbox
    stays dirty between tasks.
    """

    def test_a_clean_finish_drops_the_baseline(self, sandbox_src, monkeypatch) -> None:
        import jinx.runner as runner

        src, base = sandbox_src
        capture_baseline()
        assert base.exists()

        cleaned = []
        monkeypatch.setattr(runner, "clean_up_ipc_files", lambda: cleaned.append(True))
        runner._finish_run()

        assert cleaned == [True]
        assert not base.exists()

    def test_an_error_exit_keeps_the_baseline_for_repair(self, sandbox_src,
                                                        monkeypatch) -> None:
        """A run that dies mid-edit must leave the recovery snapshot behind.

        The bootstrap preflight can only roll a half-applied self-edit back if
        the baseline it compares against survived the crash, so the error paths
        have to be checked by actually taking one.
        """
        import jinx.runner as runner

        src, base = sandbox_src
        capture_baseline()
        assert base.exists()
        # A half-applied edit: the run is about to die with the source modified.
        (src / "tools.py").write_text("def tool_schema(:\n", encoding="utf-8")

        cleared = []
        monkeypatch.setattr(selfpatch, "clear_baseline", lambda: cleared.append(True))

        # Create a fake run state so run_file_ipc thinks it is resuming.
        run_state_path = selfpatch._repo_root() / "jinx_run_state.yaml"
        run_state_path.write_text(yaml.dump({
            "rnd": 1, "tool_depth": 0, "history": [],
            "waiting_for": "llm_generate", "min_rounds": 10,
        }), encoding="utf-8")

        # Mock the paths to point to this fake run state.
        monkeypatch.setattr(runner, "RUN_STATE_PATH", run_state_path)
        monkeypatch.setattr(runner, "RESPONSE_PATH",
                            selfpatch._repo_root() / "no_such_response.yaml")
        monkeypatch.setattr(runner, "REQUEST_PATH",
                            selfpatch._repo_root() / "no_such_request.yaml")
        monkeypatch.setattr(runner, "clean_up_ipc_files", lambda: None)

        with pytest.raises(SystemExit):
            runner.run_file_ipc(None, None)

        assert cleared == [], "an error exit must not drop the baseline"
        assert base.exists(), "the preflight has nothing to repair from otherwise"



# ==============================================================================
# Self-patch protection
# ==============================================================================


class TestVerificationInputsAreProtected:
    """The suite that judges a self-patch is itself part of what it could edit."""

    def test_the_tests_are_part_of_the_baseline(self, sandbox_src) -> None:
        src, base = sandbox_src
        repo = selfpatch._repo_root()
        tests = repo / "tests"
        tests.mkdir(parents=True)
        (tests / "test_thing.py").write_text("def test_ok():\n    assert True\n",
                                            encoding="utf-8")
        (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n",
                                            encoding="utf-8")
        assert capture_baseline() is True
        assert "repo/tests/test_thing.py" in _baseline_files_keys()
        assert "repo/pyproject.toml" in _baseline_files_keys()
        assert baseline_changed() == []

    def test_a_weakened_test_is_restored_before_verification(self, sandbox_src,
                                                             monkeypatch) -> None:
        import jinx.runner as runner

        src, _ = sandbox_src
        tests = selfpatch._repo_root() / "tests"
        tests.mkdir(parents=True)
        test_file = tests / "test_thing.py"
        original = "def test_ok():\n    assert True\n"
        test_file.write_text(original, encoding="utf-8")
        capture_baseline()

        # The model deletes the assertion instead of fixing the code.
        test_file.write_text("def test_ok():\n    assert True  # skipped\n"
                             "def test_disabled():\n    pass\n", encoding="utf-8")
        assert "repo/tests/test_thing.py" in baseline_changed()

        seen = {}

        def fake_verify(repo_root, *a, **k):
            seen["test_file"] = test_file.read_text(encoding="utf-8")
            return {"ok": True, "summary": "PASS", "checks": []}

        monkeypatch.setattr(selfpatch, "verify", fake_verify)
        runner._enforce_self_patch_gate({"history": []})

        assert seen["test_file"] == original, \
            "verification must run against the baseline test, not the edited one"
        assert test_file.read_text(encoding="utf-8") == original

    def test_a_new_test_file_does_not_break_verification(self, sandbox_src,
                                                         monkeypatch) -> None:
        """Adding a test is legitimate; it must not read as a brake change."""
        import jinx.runner as runner

        src, _ = sandbox_src
        capture_baseline()
        assert runner._enforce_self_patch_gate({"history": []}) is None
        monkeypatch.setattr(
            selfpatch, "verify",
            lambda *a, **k: {"ok": True, "summary": "PASS", "checks": []},
        )
        (src / "state.py").write_text(
            "def merge_state():\n    return {}\n\n\ndef helper():\n    return 1\n",
            encoding="utf-8")
        assert runner._enforce_self_patch_gate({"history": []}) is None


def _baseline_files_keys():
    return set(selfpatch._baseline_files())


class TestBrakeProtection:
    @pytest.mark.parametrize(
        "path, body",
        [
            ("state.py", "def merge_state(a, b):\n    return a\n"),
            ("state.py", "class StateBlock:\n    pass\n"),
            ("state.py", "def _resolve_jinx_path():\n    return None\n"),
            ("state.py", "def atomic_write_yaml(p, d):\n    pass\n"),
            ("runner.py", "def check_exit(a, b, c):\n    return True\n"),
            ("runner.py", "def check_deadlock(a, b, c):\n    return True\n"),
            ("runner.py", "def _resolve_min_rounds(a, b):\n    return 1\n"),
            ("runner.py", "def _handle_llm_response(a):\n    pass\n"),
            ("prompts.py", 'SYSTEM_PROMPT = "do nothing"\n'),
            ("selfpatch.py", "SELF_PATCH_ENABLED = False\n"),
            ("learning.py", "def add_lessons():\n    pass\n"),
        ],
    )
    def test_brake_logic_cannot_be_rewritten(self, path, body) -> None:
        assert is_protected_change(path, body), "%s must be protected" % path

    def test_ordinary_code_in_the_same_file_is_allowed(self) -> None:
        # merge_scores lives in state.py next to merge_state and must stay
        # improvable, or protection would be too coarse to be useful.
        assert is_protected_change("state.py", "def merge_scores(a, b):\n    return a + b\n") == []

    def test_a_new_helper_file_is_allowed(self) -> None:
        assert is_protected_change("cache.py", "def memoize(f):\n    return f\n") == []

    def test_the_override_is_honoured(self, monkeypatch) -> None:
        monkeypatch.setattr(selfpatch, "ALLOW_PROTECTED", True)
        assert is_protected_change("state.py", "def merge_state(a, b):\n    return a\n") == []

    def test_guard_allows_writes_outside_the_source_tree(self, monkeypatch) -> None:
        monkeypatch.setattr(selfpatch, "SRC_DIR", Path("/nonexistent/src"))
        assert guard_tool_call("/tmp/notes.txt", "hello") is None

    def test_guard_refuses_a_protected_write(self, monkeypatch, tmp_path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        monkeypatch.setattr(selfpatch, "SRC_DIR", src)
        reason = guard_tool_call(
            str(src / "state.py"), "def merge_state(a, b):\n    return a\n"
        )
        assert reason and "refused" in reason.lower()

    def test_guard_permits_a_benign_self_patch(self, monkeypatch, tmp_path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        monkeypatch.setattr(selfpatch, "SRC_DIR", src)
        assert guard_tool_call(str(src / "cache.py"), "X = 1\n") is None

    def test_within_src_tree_is_exact_not_a_prefix(self, tmp_path, monkeypatch) -> None:
        src = tmp_path / "src" / "jinx"
        src.mkdir(parents=True)
        monkeypatch.setattr(selfpatch, "SRC_DIR", src)
        assert within_src_tree(str(src / "state.py"))
        assert not within_src_tree(str(tmp_path / "src" / "jinx_other" / "state.py"))
        assert not within_src_tree(str(tmp_path))

    def test_ordinary_code_in_the_same_protected_file_is_allowed(self) -> None:
        # This is the whole point of fingerprinting bodies instead of testing for
        # the presence of a name: the model edits state.py by rewriting it whole.
        # A presence test would refuse every write to it, which would make the
        # self-patch gate useless for the files that need it most.
        from jinx import selfpatch as sp

        original = (sp.SRC_DIR / "state.py").read_text(encoding="utf-8")
        edited = original.replace(
            "FACTS_CAP", "FACTS_CAP  # widened", 1
        )
        if edited == original:
            edited = original + "\n# a harmless trailing note\n"
        assert sp.is_protected_change("state.py", edited) == [], \
            "rewriting a protected file without touching the brake must be allowed"

    def test_weakening_the_brake_in_a_full_rewrite_is_still_refused(self) -> None:
        from jinx import selfpatch as sp

        original = (sp.SRC_DIR / "state.py").read_text(encoding="utf-8")
        # A realistic attack: keep the function name, drop the validation.
        tampered = original.replace("def merge_state(", "def merge_state(  # noqa", 1)
        tampered = tampered.replace("return merged", "return merged  # unchecked", 1)
        assert sp.is_protected_change("state.py", tampered) != [], \
            "a full-file rewrite that alters merge_state must be refused"

    def test_a_new_file_defining_a_protected_symbol_is_refused(self) -> None:
        from jinx import selfpatch as sp

        assert sp.is_protected_change("state.py", "def merge_state():\n    return {}\n") != []

    def test_deleting_a_brake_is_refused(self) -> None:
        from jinx import selfpatch as sp

        original = (sp.SRC_DIR / "state.py").read_text(encoding="utf-8")
        stripped = re.sub(
            r"^def merge_state[\s\S]*?(?=\n[^\s#])", "", original, count=1, flags=re.M
        )
        assert stripped != original, "the fixture rewrite did not actually delete anything"
        assert "def merge_state" not in stripped
        assert sp.is_protected_change("state.py", stripped) != [], \
            "removing a brake outright must be refused, not merely absent"



    def test_shadowing_a_brake_with_a_later_definition_is_refused(self) -> None:
        # Python keeps the LAST definition of a name, so appending a second
        # merge_state silently replaces the validated one at import time. A guard
        # that only inspects the first match sees an untouched block and waves
        # this straight through, which is precisely the bypass it must stop.
        from jinx import selfpatch as sp

        original = (sp.SRC_DIR / "state.py").read_text(encoding="utf-8")
        shadowed = original + "\n\ndef merge_state(a, b):\n    return a  # brake removed\n"
        assert sp.is_protected_change("state.py", shadowed) != [], \
            "appending a shadowing definition must be refused"

    def test_a_moved_but_unchanged_brake_is_still_an_edit(self) -> None:
        # Same bodies, different order: the tuple comparison must notice, or a
        # reordering could be used to slip a redefinition past the check.
        from jinx import selfpatch as sp

        original = (sp.SRC_DIR / "state.py").read_text(encoding="utf-8")
        head, _, tail = original.rpartition("\ndef merge_state(")
        assert head and tail
        reordered = head + "\n" + tail.rstrip() + "\n\ndef merge_state(" + \
            original[original.rindex("\ndef merge_state(") + 1:]
        assert sp.is_protected_change("state.py", reordered) != []


class TestNestedSourcePaths:
    """Snapshots are keyed by path relative to SRC_DIR, not by basename.

    `rglob` implies subpackages, and the moment a nested module shares a filename
    with a top-level one, basename keys collide: one file silently overwrites the
    other in the snapshot, and `restore` then writes the survivor's contents to
    the source root. The rollback would land in the wrong place while the
    original stayed broken.
    """

    @pytest.fixture()
    def nested(self, tmp_path, monkeypatch):
        src = tmp_path / "src" / "jinx"
        (src / "sub").mkdir(parents=True)
        (src / "state.py").write_text("ROOT = 1\n", encoding="utf-8")
        (src / "sub" / "state.py").write_text("NESTED = 1\n", encoding="utf-8")
        monkeypatch.setattr(selfpatch, "SRC_DIR", src)
        monkeypatch.setattr(selfpatch, "BASELINE_DIR", tmp_path / "baseline")
        return src, tmp_path / "baseline"

    def test_both_files_survive_the_snapshot(self, nested) -> None:
        src, _ = nested
        snap = selfpatch.snapshot()
        assert snap == {"state.py": "ROOT = 1\n", "sub/state.py": "NESTED = 1\n"}, \
            "colliding basenames must not overwrite each other"

    def test_a_nested_file_is_restored_where_it_belongs(self, nested) -> None:
        src, base = nested
        assert capture_baseline() is True
        (src / "sub" / "state.py").write_text("NESTED = 999\n", encoding="utf-8")
        (src / "state.py").write_text("ROOT = 999\n", encoding="utf-8")

        restored = restore_baseline()

        assert sorted(restored) == ["state.py", "sub/state.py"]
        assert (src / "state.py").read_text(encoding="utf-8") == "ROOT = 1\n"
        assert (src / "sub" / "state.py").read_text(encoding="utf-8") == "NESTED = 1\n"
        assert not (src / "state.py").with_suffix(".tmp").exists()

    def test_a_nested_change_is_detected(self, nested) -> None:
        src, _ = nested
        capture_baseline()
        assert baseline_changed() == []
        (src / "sub" / "state.py").write_text("NESTED = 2\n", encoding="utf-8")
        assert baseline_changed() == ["sub/state.py"], \
            "a change in a subpackage must be compared against the right baseline"

    def test_a_nested_file_added_later_is_reverted(self, nested) -> None:
        src, _ = nested
        capture_baseline()
        (src / "sub" / "sneaky.py").write_text("X = 1\n", encoding="utf-8")
        assert baseline_changed() == ["sub/sneaky.py"]
        restore_baseline()
        assert not (src / "sub" / "sneaky.py").exists(), \
            "a file the model added must be removed, not left behind"

    def test_a_new_nested_directory_is_created_on_restore(self, nested) -> None:
        src, _ = nested
        capture_baseline()
        (src / "sub" / "state.py").unlink()
        (src / "sub").rmdir()
        assert restore_baseline() == ["sub/state.py"]
        assert (src / "sub" / "state.py").read_text(encoding="utf-8") == "NESTED = 1\n"


class TestOnDiskProtectionRecheck:
    """`guard_tool_call` only sees file_write; the gate must re-check the disk.

    The source tree is equally reachable through `bash_exec`, so a `sed -i` or a
    throwaway Python script never passes the guard. A brake weakened that way
    still passes verification when the suite stays green, and would then be
    adopted as the new baseline — the gate would have blessed the one edit it
    exists to prevent.
    """

    def test_a_brake_weakened_on_disk_is_detected(self, sandbox_src) -> None:
        original = (sandbox_src[0] / "state.py").read_text(encoding="utf-8")
        capture_baseline()
        assert selfpatch.protection_violations() == {}

        # Simulate bash_exec rewriting the file: the guard never saw this.
        (sandbox_src[0] / "state.py").write_text(
            original.replace("def merge_state():", "def merge_state():  # weakened"),
            encoding="utf-8",
        )
        assert selfpatch.baseline_changed() == ["state.py"], "the diff must be seen"
        assert selfpatch.protection_violations(), "and it must be recognised as a brake edit"

    def test_an_ordinary_change_on_disk_is_not_flagged(self, sandbox_src) -> None:
        src, _ = sandbox_src
        original = (src / "state.py").read_text(encoding="utf-8")
        capture_baseline()
        (src / "state.py").write_text(
            original + "\n\ndef helper():\n    return 1\n", encoding="utf-8"
        )
        assert selfpatch.protection_violations() == {}, \
            "improving the rest of a protected file must stay possible"

    def test_a_trailing_comment_is_not_a_brake_edit(self, sandbox_src) -> None:
        # Extracting a definition up to the next top-level line used to swallow
        # whatever followed it, so a note added after the brake looked like a
        # change to the brake itself.
        src, _ = sandbox_src
        original = (src / "state.py").read_text(encoding="utf-8")
        capture_baseline()
        (src / "state.py").write_text(original + "\n\n# a note\n", encoding="utf-8")
        assert selfpatch.protection_violations() == {}

    def test_a_protected_file_edited_on_disk_is_detected(self, sandbox_src) -> None:
        src, _ = sandbox_src
        (src / "selfpatch.py").write_text("ALLOW_PROTECTED = True\n", encoding="utf-8")
        capture_baseline()
        (src / "selfpatch.py").write_text("ALLOW_PROTECTED = True  # disarmed\n",
                                         encoding="utf-8")
        assert any("selfpatch.py" in name for name in selfpatch.protection_violations())

    def test_untouched_protected_files_do_not_block_an_ordinary_edit(self, sandbox_src) -> None:
        """A real baseline always contains the protected files themselves.

        If those are compared unconditionally, every self-patch is reported as a
        violation and the gate refuses all of them.
        """
        src, _ = sandbox_src
        for name in selfpatch.PROTECTED_FILES:
            (src / name).write_text("# protected, untouched\n", encoding="utf-8")
        capture_baseline()
        assert selfpatch.protection_violations() == {}
        (src / "tools.py").write_text("# an ordinary improvement\n", encoding="utf-8")
        assert selfpatch.protection_violations() == {}
        (src / list(selfpatch.PROTECTED_FILES)[0]).write_text(
            "# protected, now changed\n", encoding="utf-8")
        assert selfpatch.protection_violations()

    def test_a_gutted_file_is_treated_as_a_removal_not_a_fragment(self, sandbox_src) -> None:
        """The on-disk comparison knows it is looking at whole files.

        Replaced by a much shorter file, ``state.py`` must read as a removed
        brake. Inferred from the line counts it would look like a small edit that
        simply does not mention the brake, and the removal would go unreported.
        """
        src, _ = sandbox_src
        (src / "state.py").write_text(
            "def merge_state():\n    return {}\n\n\ndef other():\n    return 1\n" * 8,
            encoding="utf-8")
        capture_baseline()
        (src / "state.py").write_text("# gutted\n", encoding="utf-8")
        violations = selfpatch.protection_violations()
        assert "state.py" in violations
        assert any("removed" in reason for reason in violations["state.py"])

    def test_a_deleted_protected_module_is_reported(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        (src / "state.py").unlink()
        violations = selfpatch.protection_violations()
        assert "state.py" in violations, \
            "deleting a module that defines a brake is a removal, not an absence"

    def test_the_override_still_wins(self, sandbox_src, monkeypatch) -> None:
        src, _ = sandbox_src
        original = (src / "state.py").read_text(encoding="utf-8")
        capture_baseline()
        (src / "state.py").write_text(original + "\n\ndef merge_state():\n    return None\n",
                                      encoding="utf-8")
        assert selfpatch.protection_violations()
        monkeypatch.setattr(selfpatch, "ALLOW_PROTECTED", True)
        assert selfpatch.protection_violations() == {}


class TestGateFeedbackDelivery:
    """The gate returns its verdict; the caller places it after the tool results."""

    def test_a_failed_protection_check_refuses_rather_than_verifying(
        self, sandbox_src, monkeypatch
    ) -> None:
        """If the brake check cannot run, the patch is not judged on its word.

        Continuing would reach ``capture_baseline`` and adopt whatever is on disk
        as the new trusted reference — which is exactly the outcome the check
        exists to prevent.
        """
        import jinx.runner as runner

        src, _ = sandbox_src
        capture_baseline()
        (src / "tools.py").write_text("broken(", encoding="utf-8")

        def boom(*args, **kwargs):
            raise OSError("baseline unreadable")

        # Our new implementation doesn't call protection_violations(),
        # it calls _violations_against.
        monkeypatch.setattr(selfpatch, "_violations_against", boom)
        monkeypatch.setattr(
            runner, "_rollback_and_report",
            lambda reason: "SELF-PATCH REFUSED: %s" % reason,
        )
        verified = []
        monkeypatch.setattr(
            selfpatch, "verify",
            lambda *a, **k: verified.append(True) or {"ok": True, "summary": "PASS",
                                                     "checks": []},
        )
        message = runner._enforce_self_patch_gate({"history": []})

        assert message and "SELF-PATCH REFUSED" in message
        assert "could not be completed" in message
        assert verified == [], "a patch must never be verified after a failed brake check"

    def test_a_rollback_failure_is_reported_distinctly(self, sandbox_src,
                                                       monkeypatch) -> None:
        import jinx.runner as runner

        src, _ = sandbox_src
        capture_baseline()
        (src / "tools.py").write_text("broken(", encoding="utf-8")
        monkeypatch.setattr(
            selfpatch, "protection_violations",
            lambda: {"tools.py": ["tools.py::brake"]},
        )

        def boom():
            raise OSError("disk full")

        monkeypatch.setattr(selfpatch, "restore_baseline", boom)
        message = runner._enforce_self_patch_gate({"history": []})

        assert message and "ROLLBACK FAILED" in message, \
            "a partial rollback must not be reported as a clean refusal"
        assert "disk full" in message

    def test_the_gate_returns_the_message_instead_of_stashing_it(self, sandbox_src,
                                                                 monkeypatch) -> None:
        import jinx.runner as runner

        src, _ = sandbox_src
        capture_baseline()
        (src / "tools.py").write_text("broken(", encoding="utf-8")
        monkeypatch.setattr(
            selfpatch, "verify",
            lambda *a, **k: {"ok": False, "summary": "pytest FAIL",
                             "checks": [{"name": "pytest", "tail": "SyntaxError"}]},
        )
        run_state = {"history": []}
        message = runner._enforce_self_patch_gate(run_state)

        assert message and "SELF-PATCH REVERTED" in message
        assert run_state.get("history") == [], \
            "history must not be mutated behind the caller's back"
        assert "self_patch_feedback" not in run_state, \
            "a second delivery channel would show the model the same text twice"

    def test_a_clean_round_returns_none(self, sandbox_src, monkeypatch) -> None:
        import jinx.runner as runner

        capture_baseline()
        monkeypatch.setattr(
            selfpatch, "verify",
            lambda *a, **k: pytest.fail("verify must not run when nothing changed"),
        )
        assert runner._enforce_self_patch_gate({"history": []}) is None


class TestRefusedSelfPatchConsumesToolDepth:
    """A refused tool call is still a turn, so it must count against the cap.

    The refusal path re-issued the LLM request with the *same* tool_depth. A model
    that kept re-proposing the same forbidden write therefore never reached
    TOOL_DEPTH_CAP and looped indefinitely, which is the opposite of what the cap
    is for.
    """

    def _respond_with_a_forbidden_write(self, monkeypatch, tool_use_id):
        """Drives _handle_llm_response with one protected file_write."""
        import jinx.runner as runner

        monkeypatch.setattr(runner, "write_jinx", lambda j: None)
        monkeypatch.setattr(runner, "_enforce_self_patch_gate", lambda rs: None)
        real = selfpatch.is_protected_change
        monkeypatch.setattr(
            selfpatch, "is_protected_change",
            lambda p, t: ["state.py::brake"] if p.endswith("state.py") else real(p, t),
        )
        history = []
        response = {"content": [
            {"type": "text", "text": "editing the brake"},
            {"type": "tool_use", "id": tool_use_id, "name": "file_write",
             "input": {"path": str(selfpatch.SRC_DIR / "state.py"),
                       "content": "X = 1\n"}},
        ]}
        return runner, history, response

    def test_the_depth_advances_when_every_call_is_refused(self, tmp_path,
                                                          monkeypatch) -> None:
        import yaml

        request = tmp_path / "jinx_request.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", tmp_path / "state.yaml")
        runner, history, response = self._respond_with_a_forbidden_write(
            monkeypatch, "call-1"
        )
        runner._handle_llm_response(response, history, rnd=1, tool_depth=3, min_rounds=1,
                                    run_state={})
        persisted = yaml.safe_load((tmp_path / "state.yaml").read_text(encoding="utf-8"))
        assert persisted["tool_depth"] == 4, \
            "a refused call must still advance tool_depth"

    def test_the_cap_is_enforced_on_the_refusal_path(self, tmp_path, monkeypatch) -> None:
        import yaml

        request = tmp_path / "jinx_request.yaml"
        monkeypatch.setattr("jinx.runner.REQUEST_PATH", request)
        monkeypatch.setattr("jinx.runner.RUN_STATE_PATH", tmp_path / "state.yaml")
        runner_, history, response = self._respond_with_a_forbidden_write(
            monkeypatch, "call-2"
        )
        runner_._handle_llm_response(response, history, rnd=1, tool_depth=19, min_rounds=1,
                                     run_state={})

        persisted = yaml.safe_load((tmp_path / "state.yaml").read_text(encoding="utf-8"))
        assert persisted["tool_depth"] == 20, "the updated depth must be recorded"
        written = yaml.safe_load(request.read_text(encoding="utf-8"))
        assert written.get("tools") == [], \
            "the cap must force the no-tools recovery path"
        assert "TOOL" in str(written.get("messages", "")).upper() or \
            written.get("type") == "llm_generate"


# ==============================================================================
# Snapshot / verify / revert
# ==============================================================================



class TestSelfPatchRevertsBrokenEdits:
    def test_a_clean_edit_is_not_reported_as_changed(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        assert baseline_changed() == []

    def test_a_legitimate_edit_is_detected(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        (src / "tools.py").write_text("def tool_schema():\n    return ['v2']\n", encoding="utf-8")
        assert baseline_changed() == ["tools.py"]

    def test_a_new_file_is_detected(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        (src / "cache.py").write_text("X = 1\n", encoding="utf-8")
        assert baseline_changed() == ["cache.py"]

    def test_a_deleted_file_is_detected(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        (src / "tools.py").unlink()
        assert "tools.py" in baseline_changed()

    def test_revert_restores_a_modified_file(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        original = (src / "tools.py").read_text(encoding="utf-8")
        (src / "tools.py").write_text("broken(", encoding="utf-8")
        restored = restore_baseline()
        assert "tools.py" in restored
        assert (src / "tools.py").read_text(encoding="utf-8") == original

    def test_revert_removes_a_new_file(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        (src / "cache.py").write_text("X = 1\n", encoding="utf-8")
        restore_baseline()
        assert not (src / "cache.py").exists()

    def test_revert_restores_a_deleted_file(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        original = (src / "state.py").read_text(encoding="utf-8")
        (src / "state.py").unlink()
        restore_baseline()
        assert (src / "state.py").read_text(encoding="utf-8") == original

    def test_the_baseline_survives_a_revert(self, sandbox_src) -> None:
        # Clearing the baseline here used to disarm the gate for the rest of the
        # run: baseline_changed() reports nothing without one, and a new one is
        # only captured when a new session starts, so the second bad self-patch
        # in the same run went completely unchecked.
        src, base = sandbox_src
        capture_baseline()
        (src / "tools.py").write_text("broken(", encoding="utf-8")
        restore_baseline()
        assert base.exists(), "the recovery copy must outlive a revert"
        assert baseline_changed() == [], \
            "after a successful revert the source already matches the baseline"

    def test_a_second_bad_patch_in_the_same_run_is_still_caught(self, sandbox_src) -> None:
        src, _ = sandbox_src
        original = (src / "tools.py").read_text(encoding="utf-8")
        capture_baseline()
        for content in ("broken_one(", "broken_two("):
            (src / "tools.py").write_text(content, encoding="utf-8")
            assert baseline_changed() == ["tools.py"], \
                "the gate must still fire on a later patch in the same run"
            restore_baseline()
        assert (src / "tools.py").read_text(encoding="utf-8") == original

    def test_a_failed_restore_keeps_the_baseline(self, sandbox_src, monkeypatch) -> None:
        # If the write fails, the baseline is the only remaining copy of the
        # working source. Deleting it there would make the damage permanent.
        src, base = sandbox_src
        capture_baseline()
        (src / "tools.py").write_text("broken(", encoding="utf-8")

        real_write_text = Path.write_text
        state = {"fail": True}

        def flaky(self, *a, **k):
            if state["fail"]:
                raise OSError("disk is read-only")
            return real_write_text(self, *a, **k)

        monkeypatch.setattr(Path, "write_text", flaky)
        assert restore_baseline() == [], "the write should have failed"
        assert base.exists(), "a failed restore must not destroy the recovery copy"

        state["fail"] = False
        assert restore_baseline() == ["tools.py"], "the baseline must still work"
        assert (src / "tools.py").read_text(encoding="utf-8") == \
            "def tool_schema():\n    return []\n"

    def test_an_unchanged_round_keeps_the_baseline(self, sandbox_src) -> None:
        # The baseline is the reference for the whole run, not a one-shot
        # checkpoint: dropping it on an unchanged round would disarm the gate
        # before any tool call had a chance to break something.
        src, base = sandbox_src
        capture_baseline()
        assert baseline_changed() == []
        assert base.exists()

    def test_revert_without_a_baseline_is_a_no_op(self, sandbox_src) -> None:
        src, _ = sandbox_src
        (src / "tools.py").write_text("untouched", encoding="utf-8")
        assert restore_baseline() == []
        assert (src / "tools.py").read_text(encoding="utf-8") == "untouched"

    def test_capture_overwrites_a_previous_baseline(self, sandbox_src) -> None:
        src, _ = sandbox_src
        capture_baseline()
        (src / "tools.py").write_text("v2", encoding="utf-8")
        capture_baseline()
        assert baseline_changed() == [], "recapturing must adopt the new file as the baseline"


class TestSelfPatchVerification:
    def test_a_real_suite_failure_is_reported(self, tmp_path) -> None:
        bad = tmp_path / "test_bad.py"
        bad.write_text("def test_fails():\n    assert False\n", encoding="utf-8")
        result = selfpatch.verify(tmp_path, timeout=120)
        assert result["ok"] is False
        assert "pytest FAIL" in result["summary"]

    def test_a_real_suite_pass_is_reported(self, tmp_path) -> None:
        good = tmp_path / "test_good.py"
        good.write_text("def test_passes():\n    assert True\n", encoding="utf-8")
        result = selfpatch.verify(tmp_path, timeout=120)
        assert result["checks"][0]["ok"] is True

    def test_a_syntax_error_is_caught(self, tmp_path) -> None:
        # The failure mode that matters most: the agent writes broken Python and
        # must not be able to keep it.
        broken = tmp_path / "test_broken.py"
        broken.write_text("def test_x(:\n    pass\n", encoding="utf-8")
        assert selfpatch.verify(tmp_path, timeout=120)["ok"] is False

    def test_the_failure_tail_is_available_for_feedback(self, tmp_path) -> None:
        bad = tmp_path / "test_bad.py"
        bad.write_text("def test_fails():\n    assert False\n", encoding="utf-8")
        result = selfpatch.verify(tmp_path, timeout=120)
        assert result["checks"][0]["tail"].strip(), "the model needs the output to fix itself"

    def test_verification_stops_at_the_first_failure(self, tmp_path) -> None:
        # Otherwise jinx_test would report noise derived from the first breakage.
        bad = tmp_path / "test_bad.py"
        bad.write_text("def test_fails():\n    assert False\n", encoding="utf-8")
        result = selfpatch.verify(tmp_path, timeout=120)
        assert len(result["checks"]) == 1

# ==============================================================================
# End-to-end: a real broken self-edit is rolled back
# ==============================================================================


class TestSelfPatchEndToEnd:
    def test_a_broken_edit_to_a_copy_of_the_framework_is_reverted(self, tmp_path, monkeypatch) -> None:
        """Fault injection against a real copy of the source tree.

        Uses the genuine jinx package rather than a stub, so the revert is proven
        against the same files a self-patch would actually touch.
        """
        import shutil

        from jinx import runner

        real_src = Path(__file__).resolve().parent.parent / ".agent" / "src" / "jinx"
        if not real_src.exists():
            pytest.skip("framework source not available")

        work = tmp_path / "src" / "jinx"
        work.parent.mkdir(parents=True)
        shutil.copytree(real_src, work, ignore=shutil.ignore_patterns("__pycache__"))
        monkeypatch.setattr(selfpatch, "SRC_DIR", work)
        monkeypatch.setattr(selfpatch, "BASELINE_DIR", tmp_path / "baseline")
        monkeypatch.setattr(runner, "AGENT_DIR", tmp_path / ".agent")

        assert selfpatch.capture_baseline() is True
        assert baseline_changed() == []

        target = work / "tools.py"
        original = target.read_text(encoding="utf-8")

        # The agent "improves" the tool registry and introduces a syntax error.
        target.write_text("def tool_schema(:\n    return []\n", encoding="utf-8")
        assert baseline_changed() == ["tools.py"]

        result = selfpatch.verify(tmp_path, timeout=300)
        assert result["ok"] is False, "the gate must catch a syntax error"

        restored = restore_baseline()
        assert "tools.py" in restored
        assert target.read_text(encoding="utf-8") == original
        compile(target.read_text(encoding="utf-8"), str(target), "exec")
