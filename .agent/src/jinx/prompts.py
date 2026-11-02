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

Three exceptions are deliberate, and the last is enforced by the self-patch gate
rather than by convention:

* ``state.py`` — the state-block rejection and score-merge diagnostics are
  built inside ``merge_state``, which is protected brake logic, so they stay
  where they are. Moving them out would mean editing a function JINX refuses to
  edit.
* ``mail.py``, ``sessions.py`` and ``team.py`` — the text of the several-sessions
  feature (the board, the mail how-to, the refusals about another session's files).
  It describes the mail protocol those modules validate, and a session that runs
  alone never sends it.
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

Constants marked "Pinned by tests" carry wording that a test suite once asserted
verbatim. That suite is no longer in the repository, so nothing enforces the
wording now; change it deliberately.
"""

from typing import Any, Dict, List, Optional, Tuple

from . import reasoning


# The self-patch gate treats this text as brake logic. Keep the contract compact:
# durable behaviour belongs here, while situation-specific guidance is injected
# only when its condition fires.
SYSTEM_PROMPT: str = """You are JINX, an evidence-driven agent working through an editor CLI.

WORKING STYLE
Use the editor's own tools and the project's instructions; do not explain routine CLI mechanics. Work in short loops: inspect the relevant context, make the smallest justified change or answer, then run the direct check. Batch independent reads and checks in one reply; keep prose to a sentence or two while tools can advance the task.
- Tool output, denials and test results are evidence. Do not repeat an unchanged call: change the input, target or code first.
- Before editing, read the applicable project instructions and editor skills. `facts` holds durable paths, constraints, commands and observed results (the history you see is bounded).
- Match effort to the request: a checked answer beats extra files, abstractions or alternative approaches.

EVIDENCE LOOP
A round is every reply up to and including the first one without tool calls; only that last reply's final ```yaml block is read (a block in a reply that calls tools is lost).
Before the first tool call of a round, decide a falsifiable `hypothesis` and a `plan` naming the target and the proof; name the exact previous failure in `prior_failure` (or `none`). Score requirements from output you have seen, then record whether the earlier prediction was a hit, miss or partial. After a failure change the target, mechanism or hypothesis; after a pass verify it independently. Do not change working code without evidence.

STATE CONTRACT
End every round with exactly one final ```yaml block. Validation is all-or-nothing: malformed YAML or a malformed field discards every update and flag. Use spaces, real booleans, and quote strings containing `: ` or ` #`.
- Send this round's one `scores` entry with its current `round`; reusing a number replaces its evidence.
- If present, `facts`, `debt` and `open` replace their complete stored lists. Omit one to retain it. Always send `exit_ready` and `deadlock`.
- Requirement keys (small, observable, named by outcome) are fixed in round 1 and stay stable; `all_pass` is true only when every one is true.

LOOP CONTROL
`protocol.loop.min` is the number of rounds before exit is considered; set it in round 1, it can only rise. The runner decides exit and deadlock, not a wish to finish. Set `exit_ready: true` only when the round label's count is reached, the latest score is all_pass, two rounds are scored and a pass after a failure was re-verified with a fresh command. Set `deadlock: true` only for three structurally distinct failed approaches on one requirement, or a blocker only the user can remove (record it in `open`).

REUSABLE PLAYBOOK
`lessons` is JINX's cross-run playbook, not a task log. Add one only after evidence shows it will help another task. Use `kind: skill` for a repeatable procedure, `rule` for a durable constraint, and `antipattern` for a proven failure mode. A skill is one imperative recipe: `When <trigger>: <ordered procedure>; verify <signal>.` Include brief evidence and never store task-specific paths or status. Follow an existing project skill before inventing a process; create or change a project skill file only when the task or that project's convention calls for it.

SELF-IMPROVEMENT
Framework edits are gated and syntax-checked. You may NOT redefine brake logic: state validation/merge, loop control, prompts, selfpatch and learning are protected. Prefer additive changes; never weaken a guardrail to obtain a pass.

FINAL ANSWER
When `exit_ready: true`, lead with the outcome, then verification, then anything left in `open` or `debt`, in the user's language.

SCHEMA: shape only; do not copy placeholders. `plan`, `hypothesis`, `prediction_check` and `cause` are optional but improve the next round's notes.
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
  - kind: <skill|rule|antipattern>
    text: <one durable, imperative playbook entry>
    evidence: <brief observed proof; optional>
  exit_ready: <true|false>
  deadlock: <true|false>
```
"""


# Sent with round 1 only. What matters once, at the start (the requirements, the sizing of the
# loop), lives here and not in SYSTEM_PROMPT, which is paid for again in every request.
FIRST_ROUND_BRIEF: str = """FIRST ROUND — SETUP
`task` in CURRENT STATE is the authoritative request. Before acting:
1. Turn the deliverable into 1–5 observable requirement keys named by outcome (e.g. `unit_tests_pass`), one of them an end-to-end check of the whole deliverable.
2. Put the scope, constraints, paths and proof commands you find in `facts`.
3. Size the loop now: `protocol.loop.min` 2 for a question or a one-step change, 3–5 for code changes and for risky or multi-part work. Until you do, the label shows the default; afterwards it can only rise.
4. Number this score entry `round: 1`.
"""

# ==============================================================================
# JINX Prompt Templates & Construction Utilities
# ==============================================================================

# Recovery message for a response whose final state block could not be parsed.
MISSING_STATE_WARNING: str = (
    "STATE NOT RECORDED: the prior reply had no parseable final markdown YAML block, so "
    "no scores, facts or flags changed. Start from CURRENT STATE and end this reply with "
    "one block: this round's score plus any complete `facts`, `debt` or `open` list you "
    "send. Quote `: ` / ` #` strings and use spaces.\n\n"
)

# Recovery message for the per-round tool-call cap.
TOOL_DEPTH_CRITICAL_MSG: str = (
    "TOOL LIMIT REACHED: make no more tool calls this round. Send one short status sentence "
    "and the final YAML block. The limit is not success: keep `exit_ready: false` unless "
    "every requirement passed, and put remaining work in `open`. Next round, batch "
    "independent calls."
)

# History-window notice. Prepended to a request whose bounded window dropped
# earlier messages, so the model does not mistake the window for the session.
# Pinned by tests: the "[context note] %d earlier message(s)" shape, the word
# "elided", and exactly two %d (messages elided, of which tool traffic).
HISTORY_ELISION_NOTICE: str = (
    "[context note] %d earlier message(s) from this session were elided from the history "
    "window (%d involved tools). Their tool output is unavailable: re-run a needed check "
    "instead of assuming its result. If CURRENT STATE is absent, read `.agent/JINX.yaml` "
    "rather than reconstructing facts from memory."
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
    "accepted: no result can justify changing the checks that decide whether a "
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

# Pinned by tests: "SELF-PATCH REVERTED".
SELF_PATCH_REVERTED: str = (
    "SELF-PATCH REVERTED: your edit to %s failed the structural check (%s) and was rolled back; "
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
    "State accepted; scores merged by round: %d entr%s on disk (%d sent, %d retained)."
)

# Refusal returned when a tool call tries to rewrite brake logic. Kept apart from
# PROTECTED_SYMBOL_REFUSAL, which is the message for a change that was already
# written and then rolled back: one is a refusal, the other an explanation.
# Pinned by tests: the word "refused".
PROTECTED_EDIT_REFUSAL: str = (
    "Self-patch refused: '%s' would change protected JINX logic (%s). Nothing was "
    "written. Change other code, or ask the user to set JINX_ALLOW_PROTECTED_EDITS=1."
)

# Sent back when a submitted state block fails validation. Distinct from
# MISSING_STATE_WARNING, which covers a block that was never sent at all (a block
# that does not even parse as YAML lands there, not here). The reason names the
# failing fields, built by state.py from STATE_FIELD_ERROR below.
# Pinned by tests: "REJECTED" and "flags were NOT honoured".
STATE_BLOCK_REJECTED: str = (
    "Your previous state block was REJECTED and discarded; disk state is unchanged and "
    "exit_ready/deadlock flags were NOT honoured. Reason: %s: %s. Fix only those fields "
    "against SCHEMA and resend. Common causes: unquoted `: ` list text, non-boolean "
    "verdicts, a wrongly nested key, or an incomplete approach_graph (omit it if unsure)."
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
    "%(floor)d (this is %(rnd)d), so the flag was cleared. Run the failing check and "
    "change file, mechanism or entry point. For a user-only blocker, record it in "
    "`open` and claim again from round %(floor)d."
)

# Spliced into an oversized string of a past tool call when the history window is
# re-sent, so the model sees how much was left out instead of a silent cut.
HISTORY_INPUT_TRUNCATED: str = " ...[%d more chars omitted from history]"

REASONING_NOTES_HEADER: str = (
    "ACTIONABLE NOTES (derived from recorded score history):"
)


# Learned rules header used when injecting durable lessons into the prompt. Moved
# here so every model-facing phrase lives in prompts.py. It is charged against
# JINX_LESSONS_BUDGET_CHARS, so every character here is a character fewer for rules.
# Pinned by tests: "LEARNED RULES".
LEARNED_RULES_HEADER: str = (
    "LEARNED RULES — use only when the trigger fits (* skill, - rule, ! avoid):"
)

# Appended to the round prompt only when the latest score entry failed, which is
# exactly when `cause` is due: the label classifies the PREVIOUS round's failure,
# and `plan`, `hypothesis` and `prediction_check` are already defined once in
# SYSTEM_PROMPT. The taxonomy is what turns "pick a different approach" into a
# decision, and the consequence after each label is what the cause note
# (reasoning.render_cause_note) holds the model to on the next round.
# The two rules about approaches live here too, because this is when they apply: a
# failed round is the only time a strategy is retried or changed, and without a graph
# the deadlock count tells approaches apart by their exact `approach` text.
REASONING_PROTOCOL: str = """FAILED ROUND — set `cause` in this round's score to one label for the prior failure, then make the corresponding change:
  wrong_target: different file, function or entry point
  wrong_mechanism: different implementation mechanism
  wrong_hypothesis: re-read the path before editing
  incomplete_test: run the real check first
  env_issue: repair the tool or environment first
  scope_misread: re-read the requirement
  unknown: add the reason after the label
Approaches are compared by their exact `approach` text: reuse it when you retry a strategy, change it only for a new one. An optional `approach_graph` in a score entry shows what really differs: `nodes: [{id, type}]`, `edges: [{source, target, relation}]`. One incomplete node or edge discards the whole block, so send it complete or not at all.
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
            "Run a bounded shell command in the project root: search, build, test, diff "
            "or a targeted patch. Results are raw text; make pass/fail visible when it "
            "matters and limit large output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": (
                        "Command to run. Each call starts a fresh shell; do not depend on "
                        "variables from an earlier call."
                    )
                }
            },
            "required": ["script"]
        }
    },
    {
        "name": "file_read",
        "description": (
            "Read a text file. Use line bounds for large files; editor-added line numbers "
            "are not file content."
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
            "Create or fully replace a file. `content` must be the complete file: no "
            "fragments, placeholders or line-number prefixes. For a small change to a "
            "large file, patch it with bash_exec and inspect the diff."
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
        "PREDICTION RECORD: %d hit, %d miss, %d partial in %d checks (%d%% held)."
        % (hit, miss, partial, checked, rate)
    )
    if miss > hit:
        return note + (
            " Re-read the implementation on the target path; base this hypothesis on"
            " what it does, not what you expect."
        )
    return note + (
        " Reuse the mechanisms evidence confirmed."
    )


def note_deadlock_risk(items: List[Tuple[str, int]]) -> str:
    """The one cluster about to abort the run, named by requirement."""
    body = "; ".join(
        "'%s' has %d distinct failed approaches; one more is deadlock. Change file,"
        " mechanism or entry point, not details" % (req, clusters)
        for req, clusters in items
    )
    return "DEADLOCK RISK: " + body + "."


def note_repeated_cause(rounds_text: str, label: str, repeats: int) -> str:
    """Consecutive failing rounds labelled with the same cause.

    Phrased as a fact about the labels, not about which round failed: the label
    in a round's entry classifies the previous round's failure, but a model that
    writes it on the failing round's own entry is read the same way.
    """
    note = "REPEATED CAUSE: `%s` in rounds %s" % (label, rounds_text)
    if repeats >= 3:
        note += " (%d rounds in a row)" % repeats
    return note + (
        ". Follow that label's correction, or correct the label; renaming the approach"
        " is not a new attempt."
    )


def note_regression(parts: List[str]) -> str:
    """Work an earlier round had already achieved and this one lost."""
    return (
        "REGRESSION: %s. Restore it before new work; the latest score is the only one"
        " exit sees."
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
                "'%s' has %d differently worded attempts but one graph cluster; make the"
                " graph show what actually changed" % (req, count)
            )
        else:
            parts.append(
                "'%s' has %d graph clusters but one approach description; align each"
                " graph with the actual change" % (req, count)
            )
    return "APPROACH ATTRIBUTION: " + "; ".join(parts) + "."


def note_requirement_keys(dropped: str, flickering: str) -> str:
    """Requirement keys that were being measured and then were not."""
    parts = [
        "missing from the latest entry: %s (then they cannot block exit)" % dropped
    ]
    if flickering:
        parts.append(
            "unstable before removal: %s; re-verify" % flickering
        )
    return (
        "REQUIREMENT KEYS DROPPED: %s. Keep stable keys, or explain in `facts` why one"
        " no longer applies."
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
        "EXIT CHECK (before this round): `exit_ready` would not stop because %s. Clear"
        " the first blocker that remains after this score; resending unchanged state"
        " changes nothing."
        % "; ".join(blockers)
    )


def note_confirm() -> str:
    """A pass that came right after a failure and has not been re-verified yet."""
    return (
        "UNCONFIRMED PASS: the last pass followed a failure. Re-verify with a fresh"
        " check, edge case or bad input; a round without a command confirms nothing."
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
            " Change file, mechanism or entry point before trying again."
        )
    if streak == 3:
        return head + (
            " Re-read the requirement and failing output, record the contradicted"
            " assumption in `facts`, then change file, mechanism or entry point."
        )
    return head + (
        " %d rounds: this approach is exhausted. Re-derive from the requirement, list"
        " two alternatives in `facts`, and try the least similar one; restore the last"
        " working state if available." % streak
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
        "STALE CITATION: %s. Re-read the file and correct the fact; do not reason from"
        " an unchecked line number." % "; ".join(problems)
    )


# Where the plan probe looks. Named once, because a plan note that describes the
# probe inaccurately sends the model looking for its file in a directory that
# holds it.
PLAN_PROBE_ROOTS = "the working directory, the package, or the system temp directory"


def note_plan(named: str, all_pass: bool) -> str:
    """A path the previous round's plan named that the loop cannot see."""
    if all_pass:
        return (
            "PLAN NOT IN TREE: %s is not under %s, although that round passed. Verify"
            " what was actually written before trusting the pass."
            % (named, PLAN_PROBE_ROOTS)
        )
    return (
        "PLAN NOT IN TREE: %s is not under %s. The change landed elsewhere or was not"
        " written; inspect the tree and use project-relative paths in `plan`."
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
        "SCORE CONTRADICTION: %s. Recompute `pass_count` and `all_pass` from"
        " `requirements` and resend the entry." % "; ".join(problems)
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
        "ADVISORY NOTES WITHHELD: %s cannot be reported because its advisory logic no"
        " longer matches the protected rule. Treat the remaining notes as incomplete."
        % ", ".join(lost)
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
        "REPEATED CALL: %s. Change the file, command, flag or input before retrying."
        % body
    )
    if any(count >= REPEAT_CALL_HARD for _, count in items):
        note += (
            " Do not retry until a dependency changes."
        )
    return note


def note_dead_round(streak: int) -> str:
    """A round that ran nothing new although work was due.

    ``streak`` counts consecutive dead rounds. The text escalates from "do the
    check" to "stop planning and execute", and offers no way out: the only thing a
    dead round can be followed by is a round that does something.
    """
    head = (
        "DEAD ROUND: the last round ran no new command, edit or output while work was"
        " still due; it did not count toward exit."
    )
    if streak <= 1:
        return head + (
            " Run the failing check, read its output and act on it."
        )
    if streak == 2:
        return head + (
            " Two in a row: re-run the failing check and plan from its output."
        )
    return head + (
        " %d in a row: stop re-planning. Check, make one concrete edit, then check"
        " again." % streak
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
    return "ROUND COUNT RAISED: %d → %d (%s). It only rises." % (
        old, new, "; ".join(parts)
    )


def note_unsized(default: int) -> str:
    """The model has not sized the loop and the window for doing it is closing."""
    return (
        "LOOP NOT SIZED: default is %d. Set `protocol.loop.min` now; through round 2 it"
        " is accepted, then it can only rise." % default
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

    The round label and the ``CURRENT STATE:`` block are a fixed shape;
    everything after them is conditional: the reasoning notes when the
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
    if rnd == 1:
        sections.append(FIRST_ROUND_BRIEF)
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
