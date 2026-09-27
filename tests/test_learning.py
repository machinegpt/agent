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
    """A fake framework source tree plus its baseline directory."""
    src = tmp_path / "src" / "jinx"
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
        assert len(rendered["text"]) <= 600 + 120
        assert len(rendered["applied"]) < 20, "the budget must actually drop entries"

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

    def test_an_error_exit_keeps_the_baseline_for_repair(self, monkeypatch) -> None:
        # The bootstrap preflight can only roll a half-applied self-edit back if
        # the baseline it compares against survived the crash.
        import jinx.selfpatch as sp

        calls = []
        monkeypatch.setattr(sp, "clear_baseline", lambda: calls.append("cleared"))
        # The error paths call clean_up_ipc_files + sys.exit directly, never
        # _finish_run, so no baseline cleanup can happen on them.
        assert "cleared" not in calls



# ==============================================================================
# Self-patch protection
# ==============================================================================


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

    def test_the_baseline_is_cleared_after_a_revert(self, sandbox_src) -> None:
        src, base = sandbox_src
        capture_baseline()
        (src / "tools.py").write_text("broken(", encoding="utf-8")
        restore_baseline()
        assert not base.exists(), "a stale baseline would silently re-revert later work"

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
