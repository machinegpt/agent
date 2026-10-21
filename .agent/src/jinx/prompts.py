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
"""Prompt definitions and constructor constants for the JINX Sovereign Agent Framework.

This module is the single home for every piece of text JINX sends to the model:
the system prompt, the per-round user prompt, the tool declarations offered in
each request, and every notice, warning, diagnostic and refusal injected
mid-run. No other module hard-codes model-facing prose; they import it from
here instead, so a wording change is a one-file change and the whole prompt
contract can be reviewed in one place.

Two exceptions are deliberate, and the second is enforced by the self-patch gate
rather than by convention:

* ``state.py`` — the state-block rejection and score-merge diagnostics are
  built inside ``merge_state``, which is protected brake logic, so they stay
  where they are. Moving them out would mean editing a function JINX refuses to
  edit.
* ``selfpatch.py`` (``guard_tool_call``'s refusal text) and ``learning.py``
  (``LEARNED_RULES_HEADER``) are protected files that JINX refuses to rewrite
  at all. See ``selfpatch.PROTECTED_FILES``.

JINX runs as a child process of an editor CLI (Claude Code, OpenCode, ...), and
the text below is written for that host:

* What the editor already provides — how its own tools behave, permission
  prompts, read-before-edit, output formatting — is not restated. The prompts
  carry only what is specific to the loop: the state block, its merge and exit
  rules, and how the three declared tools map onto the editor's own.
* The system prompt travels in every ``llm_generate`` request and the host
  reads it each time, so its length is paid per request, not per run. Guidance
  that only matters after a failed round (``REASONING_PROTOCOL``) or only when
  a condition fires (the reasoning notes) is delivered at that point instead of
  up front.
* Each rule is stated once, and every claim about runner behaviour has to match
  ``runner.py`` and ``state.py``. Re-check the prose here when either changes.

Several phrases are pinned by the test suite (``tests/`` and the
``verify_prompts`` plugin); the constants that carry them say so.
"""

from typing import Any, Dict, List, Optional, Tuple

from . import reasoning


# Pinned by tests: the first line (it is the protected brake-logic anchor), the
# headings and phrases "SELF-IMPROVEMENT", "lessons", ".agent/src/jinx/*.py" and
# "You may NOT redefine the brake logic".
SYSTEM_PROMPT: str = """You are JINX, a single-agent cognitive loop. You execute tasks through disciplined iterative refinement.

ENVIRONMENT
You run inside an editor CLI (Claude Code, OpenCode). A round is every reply up to and including the first one that makes no tool calls; the runner reads the state block of that last reply, merges it into JINX.yaml and decides whether the loop continues. Nobody reads intermediate prose live, so keep it to a sentence or two.
- Only tool_use blocks change anything; describing an edit does not. The editor runs bash_exec, file_read and file_write with its own shell, read and write tools in the project root, under its permission rules. A denial or error comes back as the result: adapt, don't repeat the call.
- Put every independent call in one reply. The depth cap (20) counts replies that call tools, not calls, and fewer replies keep CURRENT STATE inside your window.
- You see only the last few messages, so keep paths, line numbers, working commands and constraints in `facts`.
- Scale to the task: a question or chat message needs a checked answer and the earliest allowed exit, not new files or extra approaches.

ROUND PROTOCOL (enforced externally)
1. PRIOR FAILURE: name exactly what the last round failed on (`prior_failure`). No silent retries.
2. PLAN, HYPOTHESIS: before the first tool call, decide `plan` (file or function to change + the command or assertion that proves it; "create <path>" for new files) and `hypothesis` ("if I <change>, <requirement> passes because <mechanism>": falsifiable, mechanism named).
3. TRY: after a failure, change the approach structurally (other files, mechanism or entry point; compare the failing rounds' `approach_graph` and `approach`); rewording is not a new approach. After a pass, don't rewrite working code: attack it (edge cases, bad input, the project's own tests) and fix only what breaks.
4. TEST: run real commands and read their output; never score from expectation, and write no state block until you have seen it. Then set `prediction_check`: hit, miss or partial, naming the belief it confirmed or refuted.
5. SCORE: pass or fail per requirement, never holistic. Define the requirements in round 1 (what "done" means, named by outcome, e.g. `unit_tests_pass`) and keep the exact keys: renaming one breaks deadlock clustering, which matches keys literally.
6. COMMIT GATE: the deliverable works end to end, not just in parts. Exit is impossible on round 1: it needs at least 2 scored rounds.

STATE BLOCK
End the round with exactly one ```yaml block, the last fenced block of the reply. It is validated whole: one bad field discards all of it, flags included. Quote strings containing ': ' or ' #', indent with spaces, use real booleans, and replace every schema placeholder with real values.
- `scores` merges by round number: send only this round's entry (a re-sent older round overwrites the stored one).
- `facts`, `debt`, `open` are replaced by what you send: send the full current list (omit a key to keep it as stored; duplicates collapse; `facts` keeps its last 60).
- Always include `exit_ready` and `deadlock`.
- `protocol.loop.min` is the number of rounds before exit is considered. Size it in round 1: until you do, the label shows the default (normally 10), and sizing is accepted through round 2. Use 2 for a question or a one-step change, 3-5 for code changes and for work that is risky or multi-part. The count only goes up: a later, lower value is ignored, and the runner adds rounds itself after a failed round or a round that ran nothing new. The round label always shows the count in force, so read it each round.

EXIT AND DEADLOCK
- `exit_ready: true` ends the run only if the round is at least the count on the label, the latest entry is all_pass, 2+ rounds are scored and pass_count has stopped rising (a rise within the last 3 rounds blocks exit, so add requirements only for a real gap). A pass right after a failed round has to be confirmed by one more round that re-verifies it with a fresh command, and a round that ran nothing new while requirements fail cannot end the run. Set it as soon as that holds. If the runner refuses, the next round prompt says why.
- `deadlock: true` counts only from the round on the label, and only if 3+ structurally different approaches failed the same requirement, or you are blocked on something only the user can supply (say what in `open`). An earlier claim is refused and the loop goes on: keep working the failing requirement.
- Repeating an approach, or a command that returns the same result, is not progress. The prompt will say so, and the next attempt must differ in file, mechanism or entry point.

APPROACH GRAPH (optional; useful once a requirement has failed twice)
`approach_graph` in a score entry lets deadlock detection tell a real change from a rewording. `nodes`: {id, type (file, tool, action or concept)}. `edges`: {source, target, relation} between node ids. Without a graph, approaches are compared by their exact `approach` string, so keep it identical when you retry the same strategy. One incomplete node or edge rejects the whole block: omit the graph unless it is complete.

SELF-IMPROVEMENT, two separate things
1. `lessons` (always safe): durable rules for future tasks, kept in a separate ledger. Each is a short, general imperative learned from what you observed, never a note about this task ("run the narrowest failing test before the full suite", not "the bug was in auth.py"). Send only NEW ones; rules that keep failing stop being shown. Don't pad.
2. Editing `.agent/src/jinx/*.py` (gated): allowed when it is the shortest path to the requirements. After a round that touches framework source the runner runs pytest and the jinx_test suite and reverts the edit if either fails; test files you add or edit are reset first. You may NOT redefine the brake logic: `merge_state`, `StateBlock`, `atomic_write_yaml`, `_resolve_jinx_path`, `_apply_protocol` (state.py); `check_exit`, `check_deadlock`, `_resolve_min_rounds`, `_handle_llm_response`, `_scores_of`, `_loop_of`, and the round-count helpers `_declared_base`, `_round_floors`, `_round_verdict`, `_close_round`, `_finish_round`, `_record_tool_results`, `_ledger_of` with their constants (runner.py); `SYSTEM_PROMPT` (prompts.py). Such writes are refused, and selfpatch.py and learning.py are off limits entirely. Prefer additive changes; never weaken an assertion to get a green suite.

FINAL ANSWER
On the reply where you set `exit_ready: true`, make the prose above the block the answer for the user, in the language of the task: the outcome first, how you verified it, then anything left in `open` or `debt`.

SCHEMA: the shape of each field, not data to copy. `plan`, `hypothesis`, `prediction_check` and `cause` are optional but feed the notes you get back.
```yaml
id: JINX
protocol:
  loop:
    min: <int>
state:
  task: <the task as you understand it>
  facts:
  - <fact, constraint, path or working command>
  scores:
  - round: <this round's number>
    approach: <short strategy name>
    prior_failure: <what the last round failed on, or none>
    plan: <what you change + the command that proves it>
    hypothesis: <if I change X, Y passes because Z>
    prediction_check: <hit|miss|partial + the belief confirmed or refuted>
    cause: <one label from the round prompt, only after a failed round>
    requirements:
      <requirement_name>: <true|false>
    pass_count: <int, the number of true requirements>
    all_pass: <true only if every requirement is true>
  debt:
  - <shortcut taken; an empty list is written as []>
  open:
  - <unresolved issue; an empty list is written as []>
  lessons:
  - <NEW durable rule; leave the key out when there is none>
  exit_ready: <true|false>
  deadlock: <true|false>
```
"""

# ==============================================================================
# JINX Prompt Templates & Construction Utilities
# ==============================================================================

# Pinned by tests: "REQUIRED markdown YAML state block".
MISSING_STATE_WARNING: str = (
    "WARNING: your last response had no parseable REQUIRED markdown YAML state block, "
    "so nothing from it was recorded: no scores, facts or exit_ready. A block that fails "
    "to parse as YAML counts as missing: quote strings containing ': ' or ' #' and indent "
    "with spaces. Start from CURRENT STATE below and end this response with the block: "
    "this round's `scores` entry only, and the full `facts`, `debt` and `open` lists.\n\n"
)

# Pinned by tests: "inner tool-calling depth limit".
TOOL_DEPTH_CRITICAL_MSG: str = (
    "CRITICAL: the inner tool-calling depth limit has been reached. No more tools this "
    "round. Reply now with one sentence of status and the complete state block "
    "(```yaml ... ```): this round's `scores` entry only, and the full `facts`, `debt` and "
    "`open` lists.\n"
    "Being cut off does not mean the task is done: set `exit_ready: true` only if every "
    "requirement genuinely passed. Otherwise set it false and put what is left in `open`, "
    "so the next round continues from an honest state. Next round, put independent calls "
    "in one reply to stay under the cap."
)

# History-window notice. Prepended to a request whose bounded window dropped
# earlier messages, so the model does not mistake the window for the session.
# Pinned by tests: the "[context note] %d earlier message(s)" shape, the word
# "elided", and exactly two %d (messages elided, of which tool traffic).
HISTORY_ELISION_NOTICE: str = (
    "[context note] %d earlier message(s) from this session were elided from the history "
    "window to bound prompt size (%d involved tool calls). Tool output is not kept in "
    "CURRENT STATE, so an elided result is gone: call the tool again instead of assuming "
    "you still have it. If CURRENT STATE is no longer in view, the task and saved state "
    "are in `.agent/JINX.yaml`: read it (never write it) rather than rebuild `facts` from memory."
)

# One failing check inside a self-patch verification report.
CHECK_FAILURE_LINE: str = "[%s] %s"

# Protection feedback prompts. Each reads as the clause after "SELF-PATCH REFUSED: ".
PROTECTED_FILE_REFUSAL: str = (
    "your edit to JINX's own source modified %s, a protected file: it holds JINX's own "
    "guardrails, so it is off limits to self-patching."
)

PROTECTED_SYMBOL_REFUSAL: str = (
    "your edit changed protected brake logic in %s (%s). It was rolled back and NOT "
    "tested: no passing test can justify changing the checks that decide whether a "
    "self-patch is safe. Change other code instead."
)

# Pinned by tests: "could not be completed".
PROTECTION_CHECK_FAILURE: str = (
    "the protected-logic check could not be completed (%s: %s), so this "
    "edit was neither verified nor accepted"
)

# Self-patch gate feedback. The gate reports the rollback separately from the
# reason, because the two can disagree: a partially failed restore leaves files
# on disk that the model believes were undone, and saying so is the only way it
# can find out.
# Pinned by tests: "ROLLBACK FAILED", and the exception text must reach the model.
SELF_PATCH_ROLLBACK_FAILED: str = (
    "SELF-PATCH REFUSED and ROLLBACK FAILED: %s\nThe rollback itself raised %s: %s. "
    "JINX's source may be damaged: stop editing framework files, set `deadlock: true` "
    "and describe what you know in `open` so the user can step in. The next run's "
    "preflight retries the repair from the baseline."
)

# Pinned by tests: "SELF-PATCH REFUSED".
SELF_PATCH_REFUSED: str = (
    "SELF-PATCH REFUSED: %s\nFiles rolled back: %s"
)

TEST_SUITE_RESTORE_FAILURE: str = (
    "the test files that judge this patch could not be restored (%s: %s), so it was "
    "not verified against a trusted suite"
)

# Reported on a patch that PASSED verification. The suite is the yardstick, so
# the model's own test edits are reverted before it runs; without this the model
# would go on believing coverage it just wrote still exists.
# Pinned by tests: a %s for the paths, and the word "restored".
TEST_FILES_RESTORED: str = (
    "SELF-PATCH VERIFIED: your framework edit was accepted.\n"
    "The tests are the yardstick, so they were restored to the baseline before the suite "
    "ran; your changes to these files are gone: %s\n"
    "Your source change stays, but any test you added or edited for it no longer exists."
)

# Pinned by tests: "SELF-PATCH REVERTED".
SELF_PATCH_REVERTED: str = (
    "SELF-PATCH REVERTED: your edit to %s failed verification (%s) and was rolled back; "
    "the framework is intact. Read the failure below; if the idea is still right, retry "
    "it smaller or more targeted, and run the failing check yourself before you finish.\n%s"
)

# Tool-result notices. Not instructions, but handed straight back to the model
# as `tool_result` content, so they belong to the same contract. Exception
# messages raised at the runner itself (IPCError, SerializationError, ...) are
# not model-facing and stay with their raisers.
MALFORMED_TOOL_BLOCK_MSG: str = (
    "Error: malformed tool_use block: it needs a string `id` and a string `name`. "
    "Nothing was executed; resend the call."
)

INVALID_TOOL_BLOCK_INPUT_MSG: str = (
    "Error: malformed tool_use block: `input` must be an object of named parameters "
    "(for example {script: ...}). Nothing was executed; resend the call."
)

FILE_SLICE_FAILURE_MSG: str = (
    "Error: could not apply start_line/end_line (%s). Use 1-indexed integers, or omit "
    "them to read the whole file."
)

# The test-evidence block. `jinx.evidence` decides which failures exist and how
# they group by cause; every phrase the model reads in the digest is assembled
# here, so the wording can be reviewed in one place.
EVIDENCE_HEADER: str = "TEST EVIDENCE (grouped by cause, not by test):"
EVIDENCE_FAILED_COUNT: str = "%d failed"
EVIDENCE_PASSED_COUNT: str = ", %d passed"
EVIDENCE_SKIPPED_COUNT: str = ", %d skipped"
EVIDENCE_CAUSE_COUNT: str = "%d distinct cause(s):"
EVIDENCE_GROUP_LINE: str = "- %s | %s | %d test(s): %s"
EVIDENCE_MORE_EXAMPLES: str = ", ..."
EVIDENCE_MORE_CAUSES: str = "- ...and %d more cause(s) not shown"
EVIDENCE_UNDERCOUNTED: str = (
    "WARNING: %s failure(s) reported but only %s parsed (the output tail was truncated), "
    "so the causes above are partial: rerun the failing tests alone to see the rest."
)
EVIDENCE_UNPARSED_WITH_TAIL: str = (
    "TEST EVIDENCE: %s failed with no parseable failure line. Raw output tail:\n%s"
)
EVIDENCE_UNPARSED: str = (
    "TEST EVIDENCE: %s failed with no parseable failure line."
)

# Marker spliced into an elided history so the model sees a gap it cannot read,
# rather than two halves of a transcript that look consecutive.
HISTORY_ELISION_MARKER: str = "... %d line(s) elided by JINX to fit the budget ..."

# Sent back when the state block was valid but carried fewer score entries than
# the loop already stored, which the merge resolves in favour of what is on disk.
# It is appended to the round prompt on nearly every round after the first, so
# it is kept to one line.
# Pinned by tests: "merged by round"; four arguments (%d, %s, %d, %d).
STATE_SCORES_MERGED: str = (
    "State accepted; scores merged by round: %d entr%s on disk (%d sent, %d kept from earlier)."
)

# Refusal returned when a tool call tries to rewrite brake logic. Kept apart from
# PROTECTED_SYMBOL_REFUSAL, which is the message for a change that was already
# written and then rolled back: one is a refusal, the other an explanation.
# Pinned by tests: the word "refused".
PROTECTED_EDIT_REFUSAL: str = (
    "Self-patch refused: '%s' would change protected JINX logic (%s). Those checks "
    "detect a broken framework, so the agent may not rewrite them. Nothing was written. "
    "Change other code instead, or have the user set JINX_ALLOW_PROTECTED_EDITS=1."
)

# Sent back when a submitted state block fails validation. Distinct from
# MISSING_STATE_WARNING, which covers a block that was never sent at all (a block
# that does not even parse as YAML lands there, not here). The reason names the
# failing fields, built by state.py from STATE_FIELD_ERROR below.
# Pinned by tests: "REJECTED" and "flags were NOT honoured".
STATE_BLOCK_REJECTED: str = (
    "Your previous state block was REJECTED and discarded; the state on disk is "
    "unchanged and your exit_ready/deadlock flags were NOT honoured. Reason: %s: %s. "
    "Fix exactly those fields against SCHEMA and re-send the block. Usual causes: a "
    "list item containing ': ' left unquoted (YAML reads it as a mapping, not text), a "
    "verdict that is not true/false, a key nested one level too deep, or an incomplete "
    "approach_graph node or edge (omit the graph if unsure)."
)

# One failing field inside the STATE_BLOCK_REJECTED reason: where, what is wrong,
# and what was actually received. `state.py` joins up to three of them with "; ".
STATE_FIELD_ERROR: str = "%s: %s (got %s)"
STATE_FIELD_ERRORS_MORE: str = "+%d more"
# The reason when the validator gave nothing to name (an exception with no fields).
STATE_REASON_UNKNOWN: str = "no further detail"

# Sent instead of honouring a `deadlock: true` that arrived before the round from
# which a deadlock may end the run. A model that is tired of a task says "deadlock"
# long before there is any evidence for it, so the claim is refused and the flag is
# cleared; the loop goes on. Filled with a dict: `rnd` (this round) and `floor` (the
# first round at which a claim counts).
DEADLOCK_REFUSED: str = (
    "DEADLOCK CLAIM NOT ACCEPTED: a deadlock does not end the run before round "
    "%(floor)d, and this was round %(rnd)d, so the flag was cleared. Keep working: run "
    "the failing check, read its output and change the approach (file, mechanism or "
    "entry point), not its wording. If something only the user can supply blocks you, "
    "say what in `open` and claim again from round %(floor)d."
)

# Spliced into an oversized string of a past tool call when the history window is
# re-sent, so the model sees how much was left out instead of a silent cut.
HISTORY_INPUT_TRUNCATED: str = " ...[%d more chars omitted from history]"

REASONING_NOTES_HEADER: str = (
    "REASONING NOTES (computed from your score history, not from this round):"
)


# Learned rules header used when injecting durable lessons into the prompt. Moved
# here so every model-facing phrase lives in prompts.py. It is charged against
# JINX_LESSONS_BUDGET_CHARS, so every character here is a character fewer for rules.
# Pinned by tests: "LEARNED RULES".
LEARNED_RULES_HEADER: str = (
    "LEARNED RULES (from earlier runs; apply them where they fit):"
)

# Appended to the round prompt only when the latest score entry failed, which is
# exactly when `cause` is due: the label classifies the PREVIOUS round's failure,
# and `plan`, `hypothesis` and `prediction_check` are already defined once in
# SYSTEM_PROMPT. The taxonomy is what turns "pick a different approach" into a
# decision, and the consequence after each label is what the cause note
# (reasoning.render_cause_note) holds the model to on the next round.
REASONING_PROTOCOL: str = """REASONING PROTOCOL - the last round failed. Set `cause` in this round's `scores` entry to ONE label for that failure; it decides what must not stay the same:
  wrong_target - right idea, wrong file, function or entry point => move the target
  wrong_mechanism - right place, but the code does not work the way you assumed => change the mechanism
  wrong_hypothesis - your prediction was refuted: your model is wrong, not your code => re-read the code before editing
  incomplete_test - possibly right, never exercised by a real command or test => run the test before touching code
  env_issue - the tool, dependency or environment failed, not the logic => fix the environment first
  scope_misread - you solved a different problem than the requirement asked => re-read the requirement
  unknown - none of these; add the reason after the label
"""

# ==============================================================================
# Tool Declarations (the `tools` field of every llm_generate request)
# ==============================================================================
# The schema is prose aimed at the model, so it lives here with the rest of the
# prompt contract. `jinx.tools.tool_schema` hands out a deep copy of it, so a
# caller mutating the result cannot corrupt the shared template.
#
# Names and parameter names are part of the host protocol (README section 1) and
# of the runner (`file_write` is guarded by its `path` and `content`); only the
# descriptions are free. They say what the editor's own tools do not: that the
# result is plain text, that `file_write` replaces the whole file, and that line
# numbers some editors prefix to a read are not part of the file.

TOOL_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "bash_exec",
        "description": (
            "Run a shell script in the project root (the editor's shell tool): tests, "
            "builds, git, search, targeted patches. The result is plain text with no "
            "status field: when pass/fail matters, print it (`cmd; echo exit=$?`), and "
            "bound big output with head, tail or grep."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": (
                        "The script to run. Each call is a fresh shell: do not rely on "
                        "variables set by an earlier call"
                    )
                }
            },
            "required": ["script"]
        }
    },
    {
        "name": "file_read",
        "description": (
            "Read a text file (the editor's read tool). Use start_line/end_line on large "
            "files. Any line numbers the editor prefixes are not file content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path, relative to the project root or absolute"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-indexed first line to read (inclusive)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional 1-indexed last line to read (inclusive)"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "file_write",
        "description": (
            "Create a file or overwrite it completely. `content` must be the entire new "
            "file: never a fragment or an '...unchanged...' placeholder, and no line-number "
            "prefixes. For a small change to a large file, patch it in place with "
            "bash_exec and review the diff."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path, relative to the project root or absolute"
                },
                "content": {
                    "type": "string",
                    "description": "The complete new content of the file"
                }
            },
            "required": ["path", "content"]
        }
    }
]


# ==============================================================================
# Reasoning-note text
# ==============================================================================
# Every string the model reads inside the REASONING NOTES block lives here, for the
# same reason the rest of this file exists: the prompt is one artefact, and prose
# split across a computation module and its caller is prose nobody can review in
# one place. `jinx.reasoning` decides *whether* a note fires and with which
# numbers; the wording of the note is assembled here and nowhere else.
#
# Each function takes already-computed data and returns the finished sentence, so
# no model-facing string is ever built by string arithmetic in the caller. An
# empty list or empty string means the note does not apply, which is how the
# reasoning side decides whether to call at all.
#
# The whole block shares one character budget (reasoning.MAX_NOTE_CHARS) and an
# over-budget note is dropped, not cut, so every note is written to be short:
# a shorter note is a later note that still fits. The notes speak in terms of what
# to do, not of the framework function that noticed the problem.

# The clause a truncated name list ends with. A list that reads as complete when
# it was cut is a small lie told to a model deciding where to look.
NOTE_MORE_ITEMS = " and %d more"


def note_name_list(names: List[str], limit: int) -> str:
    """Up to ``limit`` quoted names, saying so when the list was cut.

    Used where the note is the only place the model learns the list is partial.
    """
    shown = ", ".join("'%s'" % name for name in names[:limit])
    if len(names) > limit:
        shown += NOTE_MORE_ITEMS % (len(names) - limit)
    return shown


def note_name_list_plain(names: List[str], limit: int) -> str:
    """Up to ``limit`` quoted names, with no mention of what was left out.

    Kept separate from ``note_name_list`` because the notes that use this one
    already name a count or a bound in the same sentence, so announcing the cut
    would double-count; the behaviour is preserved rather than unified.
    """
    return ", ".join("'%s'" % name for name in names[:limit])


def note_calibration(hit: int, miss: int, partial: int, checked: int, rate: int) -> str:
    """How well the model's own predictions about this code have held.

    Only the counts reach this function, so the advice is about the whole record
    and never claims anything about the latest prediction in particular.
    """
    note = (
        "PREDICTION RECORD: %d hit, %d miss, %d partial over %d scored round(s) "
        "(%d%% of your predictions about this code held)."
        % (hit, miss, partial, checked, rate)
    )
    if miss > hit:
        return note + (
            " More were refuted than confirmed: re-read the real implementation of the"
            " path you are about to change, and base this round's hypothesis on what you"
            " just observed, not on what you expect."
        )
    return note + (
        " Your model of this code mostly holds: keep reasoning from the mechanisms that"
        " were confirmed instead of restarting from a new idea."
    )


def note_deadlock_risk(items: List[Tuple[str, int]]) -> str:
    """The one cluster about to abort the run, named by requirement."""
    body = "; ".join(
        "requirement '%s' has failed under %d structurally different approaches and"
        " one more aborts the run as a deadlock: change the category of approach"
        " (other files, mechanism or entry point) now, not the details of the last"
        " attempt" % (req, clusters)
        for req, clusters in items
    )
    return "DEADLOCK RISK: " + body + "."


def note_repeated_cause(rounds_text: str, label: str, repeats: int) -> str:
    """Consecutive failing rounds labelled with the same cause.

    Phrased as a fact about the labels, not about which round failed: the label
    in a round's entry classifies the previous round's failure, but a model that
    writes it on the failing round's own entry is read the same way.
    """
    note = "REPEATED CAUSE: `cause` is '%s' in rounds %s" % (label, rounds_text)
    if repeats >= 3:
        note += " (%d rounds in a row)" % repeats
    return note + (
        ". Rewording the approach does not change a repeated class of mistake: do what"
        " that label requires, or correct the label if it was wrong."
    )


def note_regression(parts: List[str]) -> str:
    """Work an earlier round had already achieved and this one lost."""
    return (
        "REGRESSION: %s. Restore what was lost before any new work; exit only looks"
        " at whether the latest round passes, so nothing else will flag it."
        % "; ".join(parts)
    )


def regression_part_broken(named: str) -> str:
    """A requirement that was true earlier and is false now."""
    return "%s passed in an earlier round and fails in your latest entry" % named


def regression_part_count(latest: int, best: int) -> str:
    """A lower pass count on the same requirement set."""
    return (
        "your latest round passed %d requirements against a best of %d"
        " on the same requirement set" % (latest, best)
    )


def note_attribution(items: List[Tuple[str, int, str]]) -> str:
    """Where the approach graph and the approach sentence disagree.

    ``kind`` is "one_cluster" when the graph is coarser than the prose and
    "many_clusters" when it is finer.
    """
    parts = []
    for req, count, kind in items:
        if kind == "one_cluster":
            parts.append(
                "requirement '%s' has %d differently worded attempts, but their approach"
                " graphs are so alike that they count as one approach, so deadlock"
                " detection cannot see the later attempts as new (make each graph show"
                " what actually changed)" % (req, count)
            )
        else:
            parts.append(
                "requirement '%s' has %d structurally distinct approach graphs, but your"
                " approach descriptions read as a single attempt, so the graphs overstate"
                " how different the attempts were (make each graph match what actually"
                " changed)" % (req, count)
            )
    return "APPROACH ATTRIBUTION: " + "; ".join(parts) + "."


def note_requirement_keys(dropped: str, flickering: str) -> str:
    """Requirement keys that were being measured and then were not."""
    parts = [
        "scored earlier but missing from your latest entry: %s (a dropped requirement"
        " can no longer block exit, which reads only the latest entry)" % dropped
    ]
    if flickering:
        parts.append(
            "unstable before being dropped (flipped between pass and fail): %s, so"
            " re-verify it" % flickering
        )
    return (
        "REQUIREMENT KEYS DROPPED: %s. Keep scoring every requirement under the same"
        " key each round, or say in `facts` why one no longer applies."
        % "; ".join(parts)
    )


# The conditions check_exit applies, phrased as what is missing rather than as a
# rule. They are evaluated on the history as it stands BEFORE the current round's
# entry exists, so each one is worded as a fact about the record so far; the note
# built from them says as much, so a blocker that this round's own entry will
# clear (a second scored round, a passing entry) is not mistaken for a verdict.
def exit_blocker_min_rounds(min_rounds: int, rnd: int) -> str:
    return (
        "exit is not considered before round %d and this is round %d"
        % (min_rounds, rnd)
    )


def exit_blocker_evidence(count: int) -> str:
    return (
        "exit needs 2 scored rounds in total and only %d is on record before this round"
        % count
    )


def exit_blocker_not_all_pass() -> str:
    return "the latest recorded round is not all_pass"


def exit_blocker_plateau(last3: int, prior_best: int) -> str:
    return (
        "pass_count is still rising (%d in the last three recorded rounds against a"
        " best of %d before them) and exit waits for it to plateau"
        % (last3, prior_best)
    )


def note_exit(blockers: List[str]) -> str:
    """Why a run that looks finished may not be allowed to stop."""
    return (
        "EXIT CHECK on the history so far, before this round's entry: exit_ready would"
        " not stop the run because %s. Work the first blocker that still applies once"
        " your entry is added; re-sending the same state will not change the verdict."
        % "; ".join(blockers)
    )


def note_confirm() -> str:
    """A pass that came right after a failure and has not been re-verified yet."""
    return (
        "UNCONFIRMED PASS: the last round passed right after a failed one. Exit needs"
        " one more round that re-verifies it: run a fresh check that did not produce"
        " the pass (the project's own tests, an edge case, bad input) and fix what"
        " breaks. A round that runs no command confirms nothing."
    )


def note_repeat_attempt(rounds: List[str], approach: str, streak: int) -> str:
    """Consecutive failing rounds that restated one attempt and gained nothing.

    ``streak`` is how many rounds repeat the attempt. The wording escalates with it
    and never offers a way to stop: the one thing it asks for is a different
    attempt, and each step says more precisely what "different" has to mean.
    """
    if len(rounds) <= 4:
        rounds_text = ", ".join(rounds[:-1]) + " and " + rounds[-1]
    else:
        rounds_text = "%s to %s" % (rounds[0], rounds[-1])
    head = (
        "REPEATED ATTEMPT: rounds %s used the same approach ('%s') and gained nothing."
        % (rounds_text, approach)
    )
    if streak <= 2:
        return head + (
            " Do not run it a third time: change the file, mechanism or entry point,"
            " not the details."
        )
    if streak == 3:
        return head + (
            " Before the next edit re-read the requirement and the failing output,"
            " record in `facts` the assumption that output contradicts, then pick an"
            " approach that differs in file, mechanism or entry point."
        )
    return head + (
        " %d rounds in a row: this approach is exhausted. Re-derive the plan from the"
        " requirement text, list two other mechanisms in `facts` and try the one that"
        " shares least with the rounds above; revert to the last working state if"
        " there is one." % streak
    )


# One stale citation. Three shapes, kept apart because they need different
# corrections: a file that cannot be found, a line past the end of it, and a
# symbol that has moved since the fact was written.
def citation_unreadable(path_text: str) -> str:
    return (
        "a fact cites %s, which cannot be found under the working directory, the"
        " package or the system temp directory" % path_text
    )


def citation_past_end(path_text: str, cited: int, length: int) -> str:
    return "a fact cites %s:%d but the file has only %d lines" % (path_text, cited, length)


def citation_symbol_moved(path_text: str, cited: int, symbol: str, start: int) -> str:
    return (
        "a fact cites %s:%d for %s, which now starts at line %d"
        % (path_text, cited, symbol, start)
    )


def note_citation(problems: List[str]) -> str:
    return (
        "STALE CITATION: %s. Re-read the file and correct the fact (give the path as"
        " it sits in the tree); don't reason from a line number you have not just"
        " checked." % "; ".join(problems)
    )


# Where the plan probe looks. Named once, because a plan note that describes the
# probe inaccurately sends the model looking for its file in a directory that
# holds it.
PLAN_PROBE_ROOTS = "the working directory, the package, or the system temp directory"


def note_plan(named: str, all_pass: bool) -> str:
    """A path the previous round's plan named that the loop cannot see."""
    if all_pass:
        return (
            "PLAN NOT IN TREE: your last plan named %s, which is not under %s, yet that"
            " round passed every requirement. If the pass depended on that change,"
            " check what was actually written before trusting it."
            % (named, PLAN_PROBE_ROOTS)
        )
    return (
        "PLAN NOT IN TREE: your last plan named %s, which is not under %s, so that"
        " change landed elsewhere or was never written. Check with ls or git status,"
        " and name paths relative to the project root in `plan`."
        % (named, PLAN_PROBE_ROOTS)
    )


# The three claims a score entry makes about itself, and the ways they can
# disagree. `plural` is passed in rather than derived from a count here, so the
# grammar of the sentence is decided once, next to the sentence.
def contradiction_non_bool(named: str, plural: bool) -> str:
    return (
        "the verdict%s for %s %s not true/false, and the checks read verdicts by"
        " truthiness" % ("s" if plural else "", named,
                         "are" if plural else "is")
    )


def contradiction_pass_count(claimed: int, actual: int) -> str:
    return "pass_count is %d but the requirements map holds %d true verdict(s)" % (
        claimed, actual
    )


def contradiction_all_pass_true(named: str, plural: bool) -> str:
    return "all_pass is true while %s %s false in the same entry" % (
        named, "are" if plural else "is"
    )


def contradiction_all_pass_false(total: int) -> str:
    return "all_pass is false while all %d requirement(s) in the same entry are true" % total


def note_score_contradiction(problems: List[str]) -> str:
    return (
        "SCORE CONTRADICTION: %s. Exit reads all_pass and pass_count while deadlock"
        " detection reads the requirements map, so an entry that disagrees with itself"
        " gives them different answers. Recompute pass_count and all_pass from the"
        " requirements map and resend the entry." % "; ".join(problems)
    )


def round_problem(round_number: Any, problem: str) -> str:
    """One problem, attributed to the round whose entry holds it."""
    return "round %s: %s" % (round_number, problem)


# What a drifted mirror is no longer able to tell the model. Named per key so the
# drift note says which advice is missing rather than that something is.
MIRROR_LOST_LABELS: Dict[str, str] = {
    "exit": "why the run cannot exit yet",
    "similarity": (
        "whether a requirement is close to a deadlock and whether the"
        " approach graphs match the attempts they describe"
    ),
}


def note_mirror_drift(keys: List[str]) -> str:
    lost = [MIRROR_LOST_LABELS[key] for key in keys if key in MIRROR_LOST_LABELS]
    return (
        "ADVISORY NOTES WITHHELD: %s could not be reported, because the advisory logic"
        " no longer matches the exit and deadlock rules it describes. Treat the rest of"
        " this block as incomplete this round." % ", ".join(lost)
    )


# ------------------------------------------------------------------------------
# Round-count and repetition notices. Unlike the reasoning notes above, which are
# computed from the score history alone, these report what the RUNNER observed
# while the round ran (which commands were issued, what they returned, whether
# anything new was done) and how the number of required rounds moved because of
# it. The runner decides whether each applies; the wording is assembled here.
# ------------------------------------------------------------------------------

# From this many identical results the notice stops asking and starts forbidding.
REPEAT_CALL_HARD: int = 5


def note_repeated_call(items: List[Tuple[str, int]]) -> str:
    """The same call returning the same result again, which cannot change anything.

    Appended to the tool results of the batch that produced the repeat, because
    that is the moment the model is deciding what to run next; a note in the next
    round's prompt arrives after the whole loop of identical calls has been spent.
    """
    body = "; ".join(
        "%s has returned the identical result %d times" % (label, count)
        for label, count in items[:3]
    )
    note = (
        "REPEATED CALL: %s. Running it again cannot change the outcome: change what"
        " you run (another file, command, flag or input) or change the code it tests"
        " first." % body
    )
    if any(count >= REPEAT_CALL_HARD for _, count in items):
        note += (
            " Do not issue that call again until something it depends on has changed."
        )
    return note


def note_dead_round(streak: int) -> str:
    """A round that ran nothing new although work was due.

    ``streak`` counts consecutive dead rounds. The text escalates from "do the
    check" to "stop planning and execute", and offers no way out: the only thing a
    dead round can be followed by is a round that does something.
    """
    head = (
        "DEAD ROUND: your last round ran nothing new (no new command, edit or output)"
        " while requirements were failing or a pass was unconfirmed, so it did not"
        " count toward the round count and could not end the run."
    )
    if streak <= 1:
        return head + (
            " This round: run the failing check, read its output and change what it"
            " points at."
        )
    if streak == 2:
        return head + (
            " 2 in a row: re-run the failing check now and plan from its actual"
            " output, not from the last plan."
        )
    return head + (
        " %d in a row: stop re-planning and execute. Run the failing check, read the"
        " first error, make one concrete edit, run the check again." % streak
    )


def note_round_count(
    old: int, new: int, base: int, dead: int, failed_round: int
) -> str:
    """Why the number of required rounds went up since the last prompt.

    ``base`` is the count the model sized (and only ever raised), ``dead`` how many
    rounds that ran nothing new are being charged on top of it, and
    ``failed_round`` the failed round a later pass has to be confirmed after (0 when
    that is not what sets the count). Only the terms that apply are listed.
    """
    parts = ["your plan is %d" % base]
    if dead:
        parts.append("+%d for round(s) that ran nothing new" % dead)
    if failed_round:
        parts.append(
            "a pass after the failure in round %d needs one more confirming round"
            % failed_round
        )
    return "ROUND COUNT RAISED from %d to %d (%s). It only goes up; plan for it." % (
        old, new, "; ".join(parts)
    )


def note_unsized(default: int) -> str:
    """The model has not sized the loop and the window for doing it is closing."""
    return (
        "LOOP NOT SIZED: `protocol.loop.min` was not set, so the default of %d rounds"
        " applies. Set it in this round's block: sizing is accepted through round 2,"
        " and after that the count can only be raised." % default
    )


def note_adaptive(min_rounds: int, info: Dict[str, Any]) -> str:
    """The runner-observed notices for the round about to start, or ''.

    ``info`` carries plain numbers computed by the runner: ``last_dead`` and
    ``dead_streak`` (was the round that just ended dead, and how many in a row),
    ``shown`` (the count the previous prompt displayed), ``base``, ``dead`` (rounds
    charged), ``failing`` (the round that just ended failed), ``failed_round`` and
    ``unsized``. Missing keys read as "does not apply", so a caller that knows
    nothing about them gets an empty string.

    The count is reported as raised only after a round that did not fail. After a
    failure the count moves with every round (a pass has to be confirmed after it),
    the job is the fix, and a "raised" line on every failing round would be noise.
    """
    lines: List[str] = []
    if info.get("last_dead"):
        lines.append(note_dead_round(int(info.get("dead_streak") or 1)))
    shown = info.get("shown")
    if (
        isinstance(shown, int) and not isinstance(shown, bool) and min_rounds > shown
        and not info.get("failing")
    ):
        lines.append(note_round_count(
            shown, min_rounds, int(info.get("base") or 0),
            int(info.get("dead") or 0), int(info.get("failed_round") or 0),
        ))
    if info.get("unsized"):
        lines.append(note_unsized(int(info.get("base") or min_rounds)))
    return "\n".join(lines)


def assemble_notes(header: str, parts: List[str], max_chars: int) -> str:
    """Joins the notes that fit the budget, in the order they were produced.

    An over-budget note is dropped rather than truncated, so a late note can never
    cost an urgent one its space, and an empty part never reaches the block: a
    header with no body reads as a truncated message and spends tokens saying
    nothing. The caller owns the order, because priority is a decision about what
    the model should read first.
    """
    kept: List[str] = []
    for part in parts:
        if not part:
            continue
        projected = len(header) + 1 + sum(len(p) + 1 for p in kept) + len(part)
        if projected > max_chars:
            continue
        kept.append(part)
    if not kept:
        return ""
    return "%s\n%s" % (header, "\n".join(kept))


def construct_round_prompt(
    rnd: int, min_rounds: int, state_dump: str, missing_state: bool = False,
    lessons_text: str = "", adaptive: Optional[Dict[str, Any]] = None,
) -> str:
    """Constructs the structured user prompt for a specific execution round in the cognitive loop.

    The round label and the ``CURRENT STATE:`` block are a fixed shape that the
    tests pin; everything after them is conditional: the reasoning notes when the
    history gives them something to say, the cause taxonomy only when the latest
    round failed, and the lessons block when the ledger has rules worth showing.

    Args:
        rnd (int): The current execution round index.
        min_rounds (int): The minimum configured round threshold.
        state_dump (str): The serialized YAML or JSON string representing the current state block.
        missing_state (bool): If True, prepends the missing state block warning message.
        lessons_text (str): Pre-rendered, already-bounded LEARNED RULES block from the
            durable cross-run ledger. Passed in pre-rendered so the cost bound lives
            in one place (``learning.render_lessons``) instead of being re-derived here.
        adaptive (dict | None): What the runner observed while the previous round ran
            (see ``note_adaptive``). ``min_rounds`` is already the count in force, so
            the label needs nothing more; this only explains why it moved and what a
            dead round has to be followed by. None keeps the prompt exactly as it was.

    Returns:
        str: The fully-formed, formatted user prompt string for the cognitive loop.
    """
    warning_prefix = MISSING_STATE_WARNING if missing_state else ""
    round_label = f"ROUND {rnd} (at least {min_rounds} rounds required before exit is considered)"
    sections = [f"{warning_prefix}{round_label}\nCURRENT STATE:\n{state_dump}"]
    notes = reasoning.render_notes(
        state_dump, REASONING_NOTES_HEADER, min_rounds=min_rounds, rnd=rnd,
    )
    if notes:
        sections.append(notes)
    if adaptive:
        observed = note_adaptive(min_rounds, adaptive)
        if observed:
            sections.append(observed)
    scores = reasoning.scores_from_dump(state_dump)
    if scores:
        latest = scores[-1]
        if not (isinstance(latest, dict) and latest.get("all_pass") is True):
            sections.append(REASONING_PROTOCOL)
    if lessons_text:
        sections.append(lessons_text)
    return "\n\n".join(sections)
