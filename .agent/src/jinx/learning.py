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
"""Durable, cross-run lesson ledger — JINX's first-class learning memory.

Why this exists
---------------
JINX already "learns", but only inside a single run: ``prior_failure``,
``approach_graph``, ``facts``/``debt``/``open``. All of that is per-run working
memory, and it is deliberately truncated to keep per-round cost flat — facts at
``JINX_FACTS_CAP`` (60) and ``prior_failure`` to the last 5 rounds. The net
effect is that the most expensive thing JINX learns is thrown away between
runs, so every new task starts from zero.

The ledger is a separate file (``.agent/lessons.yaml``) precisely because
``_init_new_session`` resets ``facts``/``scores``/``debt``/``open`` in
``JINX.yaml``. Starting a new task must not erase what was learned, otherwise
this store would decay at exactly the same rate as the state it replaces.

Why it is credit-assigned, not append-only
------------------------------------------
An append-only list of rules would be an unbounded structure and would
eventually reproduce the quadratic-growth defect this file sits next to. Two
things keep it flat:

1. **Dedup + cap** on insert, and a character *budget* (not just a count) on
   injection, because characters are what actually cost tokens.
2. **Outcomes.** Every round credits the lessons it was shown with the round's
   result. A rule that keeps being applied on failing rounds sinks; one that
   works rises. Rules with a negative record are not deleted, but they stop
   being injected, so the store self-prunes instead of growing.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .state import AGENT_DIR, atomic_write_yaml

logger = logging.getLogger("jinx.learning")

# The ledger deliberately lives outside JINX.yaml: that file is reset when a new
# task starts, and the whole point here is to survive that.
LESSONS_PATH: Path = Path(
    os.environ.get("JINX_LESSONS_PATH", str(AGENT_DIR / "lessons.yaml"))
)

# Ceiling on stored lessons. Oldest are evicted first, but only after the
# credit ranking below has had a chance to drop them.
LESSONS_CAP: int = int(os.environ.get("JINX_LESSONS_CAP", "40"))

# How many lessons are injected into a round prompt at most.
LESSONS_INJECT_LIMIT: int = int(os.environ.get("JINX_LESSONS_INJECT_LIMIT", "12"))

# Character budget for the injected block. A count limit alone is not a real
# bound: five 500-character rules cost more than forty 40-character ones.
LESSONS_BUDGET_CHARS: int = int(os.environ.get("JINX_LESSONS_BUDGET_CHARS", "1200"))

VALID_KINDS = ("rule", "skill", "antipattern")

# Header of the injected block. A module constant because its length is charged
# against LESSONS_BUDGET_CHARS, so the budget boundary has to be computable.
LEARNED_RULES_HEADER = (
    "LEARNED RULES (durable, carried over from earlier sessions; "
    "verified rules float up, rules that kept failing are no longer shown):"
)


def _normalize(text: str) -> str:
    """Case/punctuation/whitespace-insensitive key for near-duplicate detection.

    Reuses the same shape as ``state._normalize_fact`` so "Don't re-send scores."
    and "dont resend scores" collapse to a single lesson.
    """
    lowered = str(text).lower()
    stripped = "".join(ch for ch in lowered if ch.isalnum() or ch.isspace())
    return " ".join(stripped.split())


def normalize_lesson_text(value: Any) -> Optional[str]:
    """Coerce a model-supplied lesson into a bounded single-paragraph string.

    Returns None when there is no usable text, so callers can drop the entry
    rather than storing an empty rule that would be rendered as noise.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        # Tolerate a mapping so the model can attach evidence inline.
        value = value.get("text") or value.get("lesson") or ""
    if not isinstance(value, str):
        value = str(value)
    text = " ".join(value.split())
    if not text:
        return None
    return text[:300]


def _coerce_kind(value: Any) -> str:
    kind = str(value or "rule").strip().lower()
    if kind in VALID_KINDS:
        return kind
    if "anti" in kind or "avoid" in kind or "never" in kind or "don't" in kind:
        return "antipattern"
    if "skill" in kind or "how" in kind or "technique" in kind:
        return "skill"
    return "rule"


def _score(lesson: Dict[str, Any]) -> float:
    """Rank a lesson by observed usefulness.

    ``confirmed - failed`` is the signal; the small ``uses`` term breaks ties in
    favour of rules that have actually been exercised, and a mild prior keeps a
    brand-new rule ahead of one with a bad record.
    """
    try:
        confirmed = int(lesson.get("confirmed", 0) or 0)
        failed = int(lesson.get("failed", 0) or 0)
        uses = int(lesson.get("uses", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return confirmed - failed + min(uses, 3) * 0.25


def add_lessons(
    existing: Optional[List[Dict[str, Any]]],
    incoming: Optional[List[Any]],
    cap: int = LESSONS_CAP,
) -> List[Dict[str, Any]]:
    """Merges model-proposed lessons into the ledger.

    Near-duplicates increment the existing entry's ``seen`` counter instead of
    appending a copy, which is what stops a model that re-states its rules every
    round from growing the file without bound. New entries start with zero
    credit, so an unproven rule never outranks a proven one.
    """
    out: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}
    for raw in existing or []:
        if not isinstance(raw, dict):
            continue
        text = normalize_lesson_text(raw.get("text"))
        if not text:
            continue
        entry = dict(raw)
        entry["text"] = text
        key = _normalize(text)
        if key in index:
            index[key]["seen"] = int(index[key].get("seen", 1) or 1) + 1
            continue
        index[key] = entry
        out.append(entry)

    for raw in incoming or []:
        if isinstance(raw, dict) and raw.get("kind"):
            kind = _coerce_kind(raw.get("kind"))
            payload = raw
        else:
            kind = "rule"
            payload = {"text": raw}
        text = normalize_lesson_text(payload.get("text"))
        if not text:
            continue
        key = _normalize(text)
        if key in index:
            target = index[key]
            target["seen"] = int(target.get("seen", 1) or 1) + 1
            continue
        entry: Dict[str, Any] = {
            "kind": kind,
            "text": text,
            "seen": 1,
            "uses": 0,
            "confirmed": 0,
            "failed": 0,
        }
        evidence = normalize_lesson_text(
            payload.get("evidence") or payload.get("why") or payload.get("detail")
        )
        if evidence:
            entry["evidence"] = evidence
        index[key] = entry
        out.append(entry)

    if cap and len(out) > cap:
        # Keep the best rules, not the most recent ones. Slicing the tail would
        # silently discard proven rules the moment a run proposes several new
        # ones. Ranking is by score with the original order as the tiebreak, so
        # equally-rated entries keep the order the ledger recorded them in.
        ranked = sorted(range(len(out)), key=lambda i: (-_score(out[i]), i))
        keep = set(ranked[:cap])
        logger.debug("lesson cap %d dropped %d lowest-scored entries", cap, len(out) - cap)
        out = [entry for i, entry in enumerate(out) if i in keep]
    return out


def record_outcome(
    lessons: Optional[List[Dict[str, Any]]],
    applied_keys: Optional[List[str]] = None,
    passed: bool = False,
) -> List[Dict[str, Any]]:
    """Credits or blames the lessons that were injected into a finished round.

    ``applied_keys`` holds the normalized keys returned by :func:`render_lessons`
    (see ``applied``), so a round is blamed only on rules it was actually shown.
    Without this the ledger would be a write-only log.
    """
    keys = set(applied_keys or [])
    if not keys:
        return lessons or []
    for lesson in lessons or []:
        if not isinstance(lesson, dict):
            continue
        if _normalize(lesson.get("text", "")) not in keys:
            continue
        lesson["uses"] = int(lesson.get("uses", 0) or 0) + 1
        if passed:
            lesson["confirmed"] = int(lesson.get("confirmed", 0) or 0) + 1
        else:
            lesson["failed"] = int(lesson.get("failed", 0) or 0) + 1
    return lessons or []


def render_lessons(
    lessons: Optional[List[Dict[str, Any]]],
    limit: int = LESSONS_INJECT_LIMIT,
    budget: int = LESSONS_BUDGET_CHARS,
) -> Dict[str, Any]:
    """Renders a bounded LEARNED RULES block for the round prompt.

    Returns a dict with ``text`` (empty when there is nothing worth saying) and
    ``applied`` — the normalized keys of the lessons actually included, which
    :func:`record_outcome` later needs. Both bounds are enforced: at most
    ``limit`` entries, and the block is truncated to fit ``budget`` characters.

    A rule that has been blamed more often than it has been confirmed scores
    below zero and is withheld entirely. Re-showing it just invites the model to
    re-apply the advice that already failed; the ledger keeps it, so it can
    recover once it is confirmed again.
    """
    candidates = [l for l in (lessons or []) if isinstance(l, dict) and l.get("text")]
    candidates = [l for l in candidates if _score(l) >= 0]
    if not candidates:
        return {"text": "", "applied": []}

    ranked = sorted(candidates, key=_score, reverse=True)
    lines: List[str] = []
    applied: List[str] = []
    # The header is text the model reads, so it is charged against the same
    # budget as the rules. Counting only the rules and trimming the finished
    # string afterwards meant the last line could be cut off while its key stayed
    # in `applied` — crediting the model for a rule it never saw.
    body_len = 0
    for lesson in ranked:
        if len(lines) >= limit:
            break
        marker = {"rule": "-", "skill": "*", "antipattern": "!"}.get(
            _coerce_kind(lesson.get("kind")), "-"
        )
        entry = "%s %s" % (marker, lesson["text"])
        # The finished block is header + "\n" + "\n".join(lines): one newline
        # after the header, one between each pair of rules, and none at the end.
        projected = (
            len(LEARNED_RULES_HEADER) + 1 + body_len + len(entry) + len(lines)
        )
        if projected > budget:
            continue
        lines.append(entry)
        applied.append(_normalize(lesson["text"]))
        body_len += len(entry)

    if not lines:
        # A budget too small for even one rule is not an error; it just means
        # nothing can honestly be shown.
        return {"text": "", "applied": []}

    return {"text": "%s\n%s" % (LEARNED_RULES_HEADER, "\n".join(lines)), "applied": applied}


def load_ledger() -> Dict[str, Any]:
    """Loads the cross-run lesson ledger, tolerating absence and corruption."""
    try:
        if not LESSONS_PATH.exists():
            return {"lessons": []}
        with open(LESSONS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        logger.error("Failed to read lessons ledger at %s: %s", LESSONS_PATH, e)
        return {"lessons": []}
    if not isinstance(data, dict):
        return {"lessons": []}
    raw = data.get("lessons")
    return {"lessons": raw if isinstance(raw, list) else []}


def save_ledger(data: Dict[str, Any]) -> None:
    """Persists the ledger atomically.

    A write failure is logged, never raised: losing the learning store must not
    be able to abort a run that is otherwise making progress.
    """
    try:
        atomic_write_yaml(LESSONS_PATH, data)
    except OSError as e:
        logger.error("Failed to persist lessons ledger: %s", e)
