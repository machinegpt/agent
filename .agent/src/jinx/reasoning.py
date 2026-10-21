"""Reasoning signals derived from the score history.

The loop protocol tells the model to deduce structure from the previous rounds'
approach graphs, but everything the state block keeps is a pass/fail flag. Eleven
questions a reasoning loop has to answer are therefore unanswerable from the
state as it stands:

* Was this round's prediction about the code right, or only its outcome? A
  requirement that fails after an edit looks identical whether the model guessed
  wrong about how the code works or implemented the right idea badly.
* Did the previous round fail the same way as the one before it? A model can
  rename its approach while repeating the identical mistake, and a name is the
  one thing the deadlock check is least able to see.
* Why is a run that looks finished not being allowed to stop? ``check_exit`` has
  a plateau rule that appears in no prompt, so a refused ``exit_ready`` is silent
  and the model spends its remaining rounds guessing.
* Which requirement keys changed? Scoring and deadlock clustering both match keys
  literally, so a renamed requirement leaves both without anything noticing.
* Did the change break something an earlier round had already fixed? Nothing
  compares a requirement's history against its latest value.
* Do the model's two descriptions of its own attempt agree? The approach graph
  and the approach sentence are written together and read by different code, and
  the deadlock check believes the graph unconditionally.
* And do the advisory notes still describe the brakes they mirror? Two of them
  re-implement protected logic, and nothing in the framework compares the two
  copies.
* Is the thing the last round said it would change actually there? The plan field
  is mandated by the protocol and read by nothing, and it is the only claim in the
  state block that can be checked against anything outside it.
* Do a round's own three claims about itself agree? A score entry carries a
  requirements map, a pass_count and an all_pass, and the two protected brakes
  read them separately, so an entry can stop the run and fail a requirement at the
  same time.
* Do the facts still say something true? They are the largest thing in the state
  block, they are written from memory, and a fact that cites a file and a line is
  the only claim here the code itself can refute.
* Is the model retrying the same attempt? Deadlock detection counts structurally
  *different* approaches, so one approach repeated round after round is a single
  cluster and never trips it, and a pass that arrives right after a failure is
  indistinguishable from one that has been verified.

This module computes all eleven from the state the model itself wrote and renders
them as a bounded block for the round prompt. It adds no state, decides nothing,
and imports nothing from the rest of the package: ``runner`` imports the prompt
module, so importing the runner from here would be circular.
``check_exit`` and ``check_deadlock`` remain the only things that end a run.

The note renderers are pure functions over score dictionaries, in the same spirit
as ``evidence.py``, with two exceptions, both of them deliberate and both bounded.
The module reaches the runner through a lazy, failure-safe probe that checks the
advisory mirrors still agree with the protected logic they describe: a mirror
nobody compares is a mirror nobody maintains, and the one place a comparison can
survive the self-patch gate is here. And ``render_plan_note`` looks at the
filesystem, because the plan field is the only claim in the state block that can
be checked against anything outside it, and a loop that only reads what the model
wrote about itself cannot tell a real edit from a described one. Both probes are
written so that any failure means no note rather than no notes.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

SCHEMA: str = "jinx.reasoning/1"

# Words a model may use for a prediction verdict. Matching looks at the first
# token, so "missed - the guard runs before the write" reads as a miss instead of
# being ignored for carrying a reason along with the verdict.
HIT_WORDS = frozenset((
    "hit", "hits", "confirmed", "correct", "true", "yes", "pass", "passed", "ok",
))
MISS_WORDS = frozenset((
    "miss", "missed", "refuted", "wrong", "false", "no", "fail", "failed", "incorrect",
))
PARTIAL_WORDS = frozenset((
    "partial", "partly", "partially", "mixed", "half", "inconclusive",
))

# The failure classes REASONING_PROTOCOL asks the model to classify with. Kept
# here as well so a note can name the label it is complaining about.
CAUSE_LABELS: Tuple[str, ...] = (
    "wrong_target",
    "wrong_mechanism",
    "wrong_hypothesis",
    "incomplete_test",
    "env_issue",
    "scope_misread",
    "unknown",
)

# check_deadlock aborts a run on the third structurally distinct failing cluster
# for a requirement. The cluster just below that is the last point at which
# changing course is still free, which is where the warning belongs.
DEADLOCK_ABORT_CLUSTERS = 3

# Same threshold and same weighting as runner._are_approaches_similar, kept in
# step deliberately: a cluster counted here has to be the cluster the abort check
# will count, or the warning fires one round away from the real thing.
SIMILARITY_THRESHOLD = 0.7

# One checked prediction says nothing about calibration; two can still be a
# coincidence, so a record is only worth the tokens once two exist.
MIN_CHECKS_FOR_RECORD = 2

# Requirement names quoted in one note, so a round that renamed ten of them does
# not crowd out the other notes.
MAX_NAMES_IN_NOTE = 4

# How much of the live score history the mirror probe compares: the widest few
# requirements, and only the most recent entries of each, because that is the
# region the deadlock check is counting clusters in. Both bounds exist so the
# probe cannot grow with the length of a run.
HISTORY_REQUIREMENTS = 4
HISTORY_ENTRIES_PER_REQUIREMENT = 6

# The plan probe is the only part of this module that touches the filesystem, so
# it is the part with the hardest bounds: how many paths one plan may contribute,
# what counts as a path, and which plans are excused because a missing file is
# exactly what they said they were going to produce.
PLAN_MAX_PATHS = 3
PLAN_PATH_SUFFIXES = (
    ".py", ".md", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".txt", ".sh", ".cmd", ".ps1",
)
# Phrases, not words. An earlier version excused any plan containing "add", which
# also excused "add a note in missing.py" - the plan said it was going to add
# something to the file, not that the file was going to appear. Each phrase below
# takes a file as its object, which is the only way a path that does not exist is
# the expected state rather than a false alarm.
PLAN_NEW_FILE_PHRASES = (
    "create ", "creating ", "add a new", "add the file", "add a file", "add a module",
    "add the module", "new file", "new module", "new package", "introduce ",
    "scaffold", "generate ", "from scratch", "placeholder", "stub out",
    "empty file", "empty module", "does not exist", "doesn't exist", "not exist",
    "not there", "no such file", "absent",
)

CITATION_PATTERN = re.compile(r"([A-Za-z0-9_./\\-]+\.py):(\d+)")

# The facts are read in every round and are written from memory, so a citation is
# the only falsifiable claim the state makes about the code. Both sides of the
# walk are bounded: how many files may be opened, and how many problems one round
# is told about.
CITATION_MAX_FILES = 3
CITATION_MAX_CITATIONS = 3

# How many recent entries the self-contradiction check reads. Three, because that
# is the window check_exit takes its plateau rule from, so a contradiction outside
# it changes no brake decision. The comment is not evidence: the cases below are
# run against the protected check_exit every round and _observed_plateau_window
# derives the window it actually uses, so a change there is drift rather than a
# stale constant nobody notices.
SCORE_CONTRADICTION_WINDOW = 3

# Score histories whose verdict depends on the plateau window, used to derive that
# window from the protected code. Each row is (pass counts, round numbers, min
# rounds). The histories are shaped so that together they rule out every window
# size but three: the 4-entry case rules out 1, 2 and 4, the 5-entry case rules out
# 1, 2 and 5, the 6-entry case rules out 1, 2 and 6. A window larger than the
# longest case is out of every case's reach, so this cannot distinguish 3 from 7 -
# it is a bound on what can be observed, not a proof.
PLATEAU_WINDOW_CASES: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], int], ...] = (
    ((0, 1, 0, 0), (1, 2, 3, 4), 0),
    ((0, 0, 1, 0, 0), (1, 2, 3, 4, 5), 0),
    ((0, 0, 0, 1, 0, 0), (1, 2, 3, 4, 5, 6), 0),
)

# Character budget for the whole note block. Counted rather than sliced, so a
# note is never handed over cut off mid-sentence, and spent in priority order:
# what stops the run, then what was lost, then this round's advice, then the rest.
MAX_NOTE_CHARS = 1800


def _val(entry: Any, key: str, default: Any = None) -> Any:
    """Reads a field from a score entry given as a dict or as a model."""
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def classify_prediction(value: Any) -> str:
    """Maps a free-text verdict to 'hit', 'miss', 'partial', or '' when unusable."""
    if value is None:
        return ""
    text = " ".join(str(value).lower().split())
    if not text:
        return ""
    head = re.sub(r"[^a-z]", "", text.split(" ", 1)[0])
    if head in HIT_WORDS or text in HIT_WORDS:
        return "hit"
    if head in MISS_WORDS or text in MISS_WORDS:
        return "miss"
    if head in PARTIAL_WORDS or text in PARTIAL_WORDS:
        return "partial"
    return ""


def cause_label(value: Any) -> str:
    """Normalizes a model-written cause to one of CAUSE_LABELS, or '' if unusable.

    The label is read from free text, so "wrong_target - it is not the guard" and
    "Wrong Target" both have to land on the same label. Separators collapse to
    underscores rather than spaces, because the labels are themselves underscored
    and a space-folded string can no longer be compared against them.
    """
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    if not text:
        return ""
    head = text.split("_", 1)[0]
    if head in CAUSE_LABELS:
        return head
    for label in CAUSE_LABELS:
        if label in text:
            return label
    return ""


def _graph_of(entry: Any) -> Optional[Dict[str, Any]]:
    """The entry's approach graph as a plain dict, or None when it has none."""
    graph = _val(entry, "approach_graph")
    if graph is None:
        return None
    if hasattr(graph, "model_dump"):
        return graph.model_dump()
    return graph if isinstance(graph, dict) else None


def _node_ids(graph: Dict[str, Any]) -> Set[str]:
    raw = graph.get("nodes")
    if not isinstance(raw, list):
        return set()
    return {
        str(n.get("id", "")).strip().lower()
        for n in raw if isinstance(n, dict) and n.get("id")
    }


def _edge_keys(graph: Dict[str, Any]) -> Set[Tuple[str, str, str]]:
    raw = graph.get("edges")
    if not isinstance(raw, list):
        return set()
    return {
        (
            str(e.get("source", "")).strip().lower(),
            str(e.get("relation", "")).strip().lower(),
            str(e.get("target", "")).strip().lower(),
        )
        for e in raw if isinstance(e, dict) and e.get("source") and e.get("target")
    }


def approaches_similar(one: Any, other: Any) -> bool:
    """The same decision runner._are_approaches_similar makes, without importing it.

    The runner imports the prompt module and the prompt module imports this one,
    so the runner is unreachable from here without a cycle. The two
    implementations are kept deliberately in step: a cluster counted in this
    module has to be the cluster check_deadlock will count.
    """
    g1 = _graph_of(one)
    g2 = _graph_of(other)
    if not g1 or not g2:
        return _val(one, "approach", "") == _val(other, "approach", "")
    nodes1, nodes2 = _node_ids(g1), _node_ids(g2)
    edges1, edges2 = _edge_keys(g1), _edge_keys(g2)
    if not (nodes1 or edges1 or nodes2 or edges2):
        return _val(one, "approach", "") == _val(other, "approach", "")
    union_nodes = nodes1 | nodes2
    union_edges = edges1 | edges2
    node_sim = len(nodes1 & nodes2) / len(union_nodes) if union_nodes else None
    edge_sim = len(edges1 & edges2) / len(union_edges) if union_edges else None
    if node_sim is not None and edge_sim is not None:
        return 0.5 * node_sim + 0.5 * edge_sim >= SIMILARITY_THRESHOLD
    if node_sim is not None:
        return node_sim >= SIMILARITY_THRESHOLD
    if edge_sim is not None:
        return edge_sim >= SIMILARITY_THRESHOLD
    return False


def cluster_count(entries: List[Any]) -> int:
    """How many structurally distinct approaches a list of entries represents."""
    clusters: List[List[Any]] = []
    for entry in entries or []:
        for cluster in clusters:
            if any(approaches_similar(entry, member) for member in cluster):
                cluster.append(entry)
                break
        else:
            clusters.append([entry])
    return len(clusters)


def _pass_count(entry: Any) -> int:
    try:
        return int(_val(entry, "pass_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _requirements(entry: Any) -> Dict[str, Any]:
    reqs = _val(entry, "requirements")
    return reqs if isinstance(reqs, dict) else {}


def _key_set(entry: Any) -> Set[str]:
    return {str(k) for k in _requirements(entry)}


def prediction_stats(scores: List[Any]) -> Dict[str, int]:
    """Counts the checked predictions in a score history."""
    hit = miss = partial = 0
    for entry in scores or []:
        verdict = classify_prediction(_val(entry, "prediction_check"))
        if verdict == "hit":
            hit += 1
        elif verdict == "miss":
            miss += 1
        elif verdict == "partial":
            partial += 1
    return {
        "hit": hit,
        "miss": miss,
        "partial": partial,
        "checked": hit + miss + partial,
    }


def _requirement_history(scores: List[Any]) -> Dict[str, List[Tuple[Any, bool]]]:
    """Per requirement, the (round, passed) observations in chronological order."""
    hist: Dict[str, List[Tuple[Any, bool]]] = {}
    for entry in scores or []:
        rnd = _val(entry, "round", "?")
        for req, passed in _requirements(entry).items():
            hist.setdefault(str(req), []).append((rnd, bool(passed)))
    return hist


def failing_entries(scores: List[Any], req: str) -> List[Any]:
    """Every entry in which ``req`` was recorded as failing."""
    return [entry for entry in scores or [] if _requirements(entry).get(req) is False]


def render_calibration_note(scores: List[Any]) -> str:
    """Tells the model how well its own predictions about this code have held.

    Silent below two checked rounds on purpose: a single hit is as likely to be
    luck as understanding, and announcing luck as calibration is worse than
    saying nothing. The counts are computed here; the sentence is prompts'.
    """
    stats = prediction_stats(scores)
    checked = stats["checked"]
    if checked < MIN_CHECKS_FOR_RECORD:
        return ""
    text = _prompts()
    if text is None:
        return ""
    return text.note_calibration(
        stats["hit"], stats["miss"], stats["partial"], checked,
        int(round(100.0 * stats["hit"] / checked)),
    )


def render_deadlock_risk_note(scores: List[Any]) -> str:
    """Warns one cluster before check_deadlock aborts the run.

    The abort needs three structurally distinct approaches to have failed on the
    same requirement, and says nothing at two. This is the warning for the round
    where changing category is still free, and it names the requirement so the
    advice is attached to the thing that is stuck.
    """
    text = _prompts()
    if text is None:
        return ""
    items: List[Tuple[str, int]] = []
    for req, entries in _failing_by_requirement(scores).items():
        clusters = cluster_count(entries)
        if clusters == DEADLOCK_ABORT_CLUSTERS - 1:
            items.append((req, clusters))
    if not items:
        return ""
    return text.note_deadlock_risk(items)


def render_cause_note(scores: List[Any]) -> str:
    """Catches a repeated failure class, which a reworded approach name hides.

    Deadlock detection compares structure, and a model that fails the same way
    twice while renaming its approach can look structurally new. The cause label
    is the one place the class of the mistake is recorded, so the same label in
    two consecutive failing rounds is the signal that the class did not change.
    """
    history = scores or []
    if len(history) < 2:
        return ""

    trail: List[Any] = []
    last_label = None
    for entry in reversed(history):
        label = cause_label(_val(entry, "cause"))
        if not label:
            break
        if last_label is None:
            last_label = label
            trail.append(entry)
            continue
        if label != last_label:
            break
        trail.append(entry)
    if len(trail) < 2:
        return ""

    run = list(reversed(trail))
    pair = run[-2:]
    if any(_val(e, "all_pass", False) for e in pair):
        return ""
    label = cause_label(_val(pair[-1], "cause"))
    if not label or cause_label(_val(pair[0], "cause")) != label:
        return ""
    text = _prompts()
    if text is None:
        return ""
    repeats = len(run)
    return text.note_repeated_cause(
        " and ".join(str(_val(e, "round", "?")) for e in pair), label, repeats
    )


def render_regression_note(scores: List[Any]) -> str:
    """Names work that an earlier round had already achieved and this one lost.

    Two signals, because the loop checks neither. ``check_exit`` reads
    ``all_pass`` of the latest entry and ``check_deadlock`` counts failing
    clusters, so a requirement that went from true to false moves no number the
    model can see; and a lower ``pass_count`` is only a regression when the round
    measured the same requirement set as the round it is being compared against.
    """
    entries = [e for e in scores or [] if _requirements(e)]
    if len(entries) < 2:
        return ""
    latest = entries[-1]
    latest_reqs = _requirements(latest)
    latest_keys = {str(k) for k in latest_reqs}
    text = _prompts()
    if text is None:
        return ""
    parts: List[str] = []

    broken = [
        req for req, obs in sorted(_requirement_history(entries[:-1]).items())
        if req in latest_reqs and any(p for _, p in obs) and latest_reqs[req] is False
    ]
    if broken:
        parts.append(text.regression_part_broken(
            text.note_name_list_plain(broken, MAX_NAMES_IN_NOTE)))

    # pass_count is only comparable against a round that measured the same keys.
    same_set = [_pass_count(e) for e in entries[:-1] if _key_set(e) == latest_keys]
    if same_set:
        best = max(same_set)
        if _pass_count(latest) < best:
            parts.append(text.regression_part_count(_pass_count(latest), best))
    if not parts:
        return ""
    return text.note_regression(parts)


def _failing_by_requirement(scores: List[Any]) -> Dict[str, List[Any]]:
    reqs: Dict[str, List[Any]] = {}
    for entry in scores or []:
        for req, passed in _requirements(entry).items():
            if not passed:
                reqs.setdefault(str(req), []).append(entry)
    return reqs


def _approach_fingerprint(entry: Any) -> str:
    """An approach sentence reduced to a bag of its content words.

    Case and word order are dropped, because the question is whether two rounds
    described the same attempt in different words, and a word-set answers that
    without pretending to understand the sentence. One- and two-letter words are
    dropped as noise, and that is the only filtering: a filler word like "then"
    does distinguish two descriptions. That limit is stated rather than hidden,
    because the note only ever fires on a disagreement, and a word of difference
    can move a pair from agreement to disagreement.
    """
    words = [w for w in re.findall(r"[a-z]+", str(_val(entry, "approach", "") or "").lower())
             if len(w) > 2]
    return " ".join(sorted(set(words)))


def render_attribution_note(scores: List[Any]) -> str:
    """Catches the model's two descriptions of its own approach disagreeing.

    The approach graph and the approach sentence are written in the same response
    and are meant to describe the same attempt. ``check_deadlock`` clusters on the
    graph alone, so when the graph claims more variety than the sentences do, the
    abort is counting attempts the model itself cannot tell apart, and three
    rewrites of one idea will abort the run as a deadlock; when the graph claims
    less, a genuinely different third attempt is folded into the first cluster and
    the abort never fires. Agreement is the normal case and stays silent.
    """
    text = _prompts()
    if text is None:
        return ""
    items: List[Tuple[str, int, str]] = []
    for req, entries in sorted(_failing_by_requirement(scores).items()):
        if len(entries) < 2:
            continue
        prints = {_approach_fingerprint(e) for e in entries}
        prints.discard("")
        if not prints:
            continue
        clusters = cluster_count(entries)
        if clusters == 1 and len(prints) > 1:
            items.append((req, len(prints), "one_cluster"))
        elif clusters > 1 and len(prints) == 1:
            items.append((req, clusters, "many_clusters"))
    if not items:
        return ""
    return text.note_attribution(items)


def _approach_text(entry: Any) -> str:
    """The entry's approach sentence, lower-cased and whitespace-collapsed."""
    return " ".join(str(_val(entry, "approach", "") or "").lower().split())


def _same_attempt(one: Any, other: Any) -> bool:
    """Whether two entries describe the same attempt, for repetition purposes.

    Wider than the deadlock check's notion in one deliberate way. ``check_deadlock``
    counts *structurally different* approaches, so a model that retries one idea
    forever is a single cluster and never trips it; this is the question that does
    catch it. Two graphs are compared by the shared similarity rule; without graphs
    the sentences are compared, exactly or as a bag of content words, so a retry
    that only reorders or recases the same words is still the same attempt. An
    entry that never named its approach ("unspecified" is the default the schema
    fills in) is not evidence of anything and never matches.
    """
    if _graph_of(one) and _graph_of(other):
        return approaches_similar(one, other)
    first, second = _approach_text(one), _approach_text(other)
    if first in ("", "unspecified") or second in ("", "unspecified"):
        return False
    return first == second or _approach_fingerprint(one) == _approach_fingerprint(other)


def repeat_streak(scores: List[Any]) -> List[Any]:
    """The trailing failing entries that restate one attempt without gaining anything.

    Walks back from the latest entry while the entries before it describe the same
    attempt and passed no fewer requirements than the latest one did, i.e. nothing
    was gained between them and now. A pass ends the walk: a repeat is only a
    problem while the requirements are failing. Fewer than two entries is not a
    repeat, so the result is ``[]`` rather than a streak of one.
    """
    trail: List[Any] = []
    for entry in reversed(list(scores or [])):
        if _val(entry, "all_pass", False):
            break
        trail.append(entry)
    trail.reverse()
    if len(trail) < 2:
        return []
    latest = trail[-1]
    streak = [latest]
    for earlier in reversed(trail[:-1]):
        if _same_attempt(earlier, latest) and _pass_count(earlier) >= _pass_count(latest):
            streak.insert(0, earlier)
        else:
            break
    return streak if len(streak) >= 2 else []


def render_repeat_note(scores: List[Any]) -> str:
    """Tells the model it is retrying the same attempt, and that this cannot work.

    Neither brake sees this. ``check_deadlock`` aborts on three structurally
    *different* failed approaches, so one approach repeated for thirty rounds is a
    single cluster and burns the whole round budget without a word being said; the
    repeated-cause note needs the model to have labelled each failure the same way.
    What is observable without either is the history itself: the same approach, no
    more requirements passing than before, round after round.

    The note escalates with the length of the streak (see the prompt module) and
    only ever asks for a different attempt: it never offers a way to stop.
    """
    streak = repeat_streak(scores)
    if not streak:
        return ""
    text = _prompts()
    if text is None:
        return ""
    return text.note_repeat_attempt(
        [str(_val(e, "round", "?")) for e in streak],
        _approach_text(streak[-1])[:60],
        len(streak),
    )


def render_confirm_note(scores: List[Any]) -> str:
    """Says that a pass which followed a failure is not trusted until re-verified.

    The runner enforces the extra round (see ``runner._round_floors``); this tells
    the model what that round is for, so it does not spend it re-stamping the pass.
    Derived from the score list alone: the latest entry passes and the one before it
    did not.
    """
    entries = list(scores or [])
    if len(entries) < 2:
        return ""
    if not _val(entries[-1], "all_pass", False) or _val(entries[-2], "all_pass", False):
        return ""
    text = _prompts()
    return text.note_confirm() if text is not None else ""


def render_requirement_note(scores: List[Any]) -> str:
    """Reports requirement keys that were being measured and then were not.

    Scoring compares literal keys and check_deadlock clusters failures by literal
    key, so a key that disappears silently leaves both: its earlier passes stop
    counting toward exit and its earlier failures stop feeding the abort check.

    A key seen once and gone is reported never. An improving round is *expected*
    to introduce requirement keys the previous round never had, so a note that
    complained about every addition would fire on nearly every round and be
    ignored within a few. The two patterns that are not expected are a key that
    was measured across several rounds and then dropped, and a key whose recorded
    verdicts flip between rounds - the second splits one requirement's failure
    history into two clusters that neither the exit check nor the deadlock check
    will ever add back together.
    """
    entries = [e for e in scores or [] if _requirements(e)]
    if len(entries) < 2:
        return ""
    latest_keys = _key_set(entries[-1])
    dropped: List[str] = []
    flickering: List[str] = []
    for req, obs in sorted(_requirement_history(entries[:-1]).items()):
        if req in latest_keys or len(obs) < 2:
            continue
        dropped.append(req)
        verdicts = [p for _, p in obs]
        if any(verdicts) and not all(verdicts):
            flickering.append(req)
    if not dropped:
        return ""
    text = _prompts()
    if text is None:
        return ""
    return text.note_requirement_keys(
        text.note_name_list_plain(dropped, MAX_NAMES_IN_NOTE),
        text.note_name_list_plain(flickering, MAX_NAMES_IN_NOTE) if flickering else "",
    )


def exit_blockers(scores: List[Any], min_rounds: int, rnd: int) -> List[str]:
    """Every condition runner.check_exit applies, phrased as what is missing.

    Advisory only: ``check_exit`` is protected brake logic and stays the only
    thing consulted. This exists because the plateau rule appears in no prompt, so
    a run that is refused for improving keeps being refused with no explanation.
    The mirror is asserted to agree with the original rather than trusted.

    Each condition is named here and phrased in the prompt module, so the rule
    stays reviewable in one place. If that module cannot be reached the list comes
    back empty, which the exit probe reads as "no blockers" and the table then
    reports as drift - the safe direction, since a note that cannot be worded
    should not be presented as advice.
    """
    text = _prompts()
    if text is None:
        return []
    blockers: List[str] = []
    if rnd < min_rounds:
        blockers.append(text.exit_blocker_min_rounds(min_rounds, rnd))
    if not scores or len(scores) < 2:
        blockers.append(text.exit_blocker_evidence(len(scores or [])))
    if scores:
        if not _val(scores[-1], "all_pass", False):
            blockers.append(text.exit_blocker_not_all_pass())
        if len(scores) >= 4:
            last3 = max(_pass_count(s) for s in scores[-3:])
            prior = scores[:-3]
            if prior:
                prior_best = max(_pass_count(s) for s in prior)
                if last3 > prior_best:
                    blockers.append(text.exit_blocker_plateau(last3, prior_best))
    return blockers


def render_exit_note(
    scores: List[Any], state: Dict[str, Any], min_rounds: int, rnd: int
) -> str:
    """Says why a run that looks finished will not be allowed to stop.

    Only for a round that claims completion (all_pass) or asks to exit, so an
    ordinary mid-task round pays nothing for this.
    """
    looks_done = bool(scores) and bool(_val(scores[-1], "all_pass", False))
    if not (looks_done or bool(state.get("exit_ready"))):
        return ""
    blockers = exit_blockers(scores, min_rounds, rnd)
    text = _prompts()
    if text is None or not blockers:
        return ""
    return text.note_exit(blockers)


def state_from_dump(state_dump: str) -> Dict[str, Any]:
    """Best-effort read of the state the round prompt already serializes.

    The dump is data the model can see, so reading it costs no new plumbing at
    the call site. Any failure yields an empty dict: these notes are an aid, and
    an aid that breaks the round prompt is worse than no aid.
    """
    try:
        data = yaml.safe_load(state_dump)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("state"), dict):
        data = data["state"]
    return data


def scores_from_dump(state_dump: str) -> List[Any]:
    """The score list out of the serialized state, or [] when there is none."""
    scores = state_from_dump(state_dump).get("scores")
    return scores if isinstance(scores, list) else []


def _plan_candidates(text: Any) -> List[str]:
    """Path-looking tokens from a plan sentence, capped and filtered.

    The protocol asks for the file or function to be changed, so a plan usually
    carries one path and often a bare identifier beside it. Only tokens that look
    like paths are considered, and the ones that cannot be a path - a URL, a glob,
    a traversal, an absolute path, a two-character bare name - are dropped rather
    than checked, because a path reported missing when it was never a path is
    worse than no path reported at all.
    """
    if not text:
        return []
    body = str(text)
    # A URL is not a path this probe can check, and the part after "://" reads
    # exactly like one, so URLs go before tokenizing rather than after. A glob
    # goes too, and for the same reason with a second symptom: "*" is not in the
    # character class below, so "src/**/*.py" would otherwise arrive as the two
    # candidates "src" and ".py" and be reported as two missing files.
    body = re.sub(r"[A-Za-z][A-Za-z0-9+.-]*://\S*", " ", body)
    body = re.sub(r"[\w.\\/]*[*?][\w.\\/*?]*", " ", body)
    roots = _probe_roots()
    found: List[str] = []
    seen: Set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9_./\\-]+", body):
        # A leading dot is part of the name, not noise: ".agent/src/..." is a real
        # path in this repo, and stripping it would report a file that is sitting
        # right there as missing. Only a "./" prefix and trailing separators go.
        token = raw.rstrip("./\\-")
        if token[:2] in ("./", ".\\"):
            token = token[2:].rstrip("./\\-")
        if not token or token in seen or len(token) < 3:
            continue
        if ".." in token:
            continue
        if raw[0] in "/\\" or re.match(r"^[A-Za-z]:", raw):
            continue

        suffix = next((s for s in PLAN_PATH_SUFFIXES if token.lower().endswith(s)), "")
        has_known_suffix = bool(suffix and len(token) > len(suffix))
        has_path_separator = "/" in token or "\\" in token

        if not has_known_suffix:
            if not has_path_separator:
                continue
            first = re.split(r"[/\\]", token, maxsplit=1)[0]
            if not first:
                continue
            if not any((root / first).is_dir() for root in roots):
                continue

        seen.add(token)
        found.append(token)
        if len(found) >= PLAN_MAX_PATHS:
            break
    return found


def _probe_roots() -> List[Any]:
    """Directories a repo-relative path is looked up under, best first.

    The working directory because that is where a repo-relative path resolves, the
    package root because the loop can be started from anywhere, the module's own
    directory because a bare name in a plan or a fact - 'runner.py' - means the
    module of ours with that name and nothing else resolves it, and the system
    temp directory because a plan may legitimately name a scratch file. This is
    where the probe stops: a path that exists somewhere else on the machine is
    invisible to it, which is why the note says where it looked rather than
    claiming the file does not exist.
    """
    roots: List[Any] = []
    for candidate in (lambda: Path.cwd(),
                      lambda: Path(__file__).resolve().parents[3],
                      lambda: Path(__file__).resolve().parent,
                      lambda: Path(tempfile.gettempdir())):
        try:
            root = candidate()
        except Exception:
            continue
        if root and root not in roots:
            roots.append(root)
    return roots


def _missing_paths(text: Any) -> List[str]:
    """Plan-named paths absent from every probe root. Empty list on any failure."""
    lowered = " ".join(str(text or "").lower().split())
    if any(phrase in lowered for phrase in PLAN_NEW_FILE_PHRASES):
        return []
    try:
        roots = _probe_roots()
    except Exception:
        return []
    if not roots:
        return []
    missing: List[str] = []
    for name in _plan_candidates(text):
        try:
            if any((root / name).exists() for root in roots):
                continue
        except Exception:
            continue
        missing.append(name)
    return missing


def _read_lines(path_text: str) -> Optional[List[str]]:
    """Lines of a file named by the state, or None when no root holds it.

    Reuses the plan probe's roots, so a fact and a plan are looked for in the same
    places and the two notes cannot disagree about where the code lives.
    Any failure is None: a citation that cannot be checked is not a stale
    citation.
    """
    try:
        for root in _probe_roots():
            target = root / path_text
            if not target.is_file():
                continue
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                return handle.read().splitlines()
    except Exception:
        return None
    return None


def _cited_symbol(prefix: str) -> str:
    """The identifier a citation points at, from text like 'check_exit (runner.py'.

    Empty when the citation is not preceded by a name, which is the common case
    for a bare 'see runner.py:202' - there is no symbol to compare a line against,
    so there is nothing to report beyond the file's length.
    """
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(?\s*$", prefix)
    return match.group(1) if match else ""


def _citation_problems(state: Dict[str, Any]) -> List[str]:
    """Claims in the state's facts that the code no longer backs, in order.

    Facts are the only place this loop stores what it knows, they are read in
    every round, and they are written by the model from memory. A fact that
    cites runner.py:202 is a falsifiable claim about a file, and it is the one
    kind of claim in the state that can be checked without the model.

    Bounded on both sides: at most CITATION_MAX_FILES files are opened and at most
    CITATION_MAX_CITATIONS problems are reported, and the first file that cannot
    be read ends the walk rather than producing a guess.
    """
    facts = state.get("facts")
    if not isinstance(facts, list):
        return []
    text = _prompts()
    if text is None:
        return []
    problems: List[str] = []
    seen: Set[str] = set()
    cache: Dict[str, Optional[List[str]]] = {}
    spent: Set[str] = set()

    def add(message: str) -> None:
        if message not in seen:
            seen.add(message)
            problems.append(message)

    for fact in facts:
        if not isinstance(fact, str) or len(problems) >= CITATION_MAX_CITATIONS:
            continue
        for path_text, digits in CITATION_PATTERN.findall(fact):
            if len(problems) >= CITATION_MAX_CITATIONS:
                break
            if ".." in path_text or path_text[0] in "/\\" or \
                    re.match(r"^[A-Za-z]:", path_text):
                # A citation that climbs out of the roots is not a citation this
                # loop will follow, whatever the file behind it is. Same rule as
                # the plan probe, so the two walks cannot disagree about it.
                continue
            if path_text in spent:
                continue
            if path_text not in cache:
                if len(cache) >= CITATION_MAX_FILES:
                    break
                cache[path_text] = _read_lines(path_text)
            lines = cache[path_text]
            prefix = fact[:fact.index("%s:%s" % (path_text, digits))]
            if lines is None:
                # Nothing else can be learned about a file that cannot be read, so
                # the rest of this round's citations of it are not walked.
                spent.add(path_text)
                add(text.citation_unreadable(path_text))
                continue
            number = _cited_symbol(prefix)
            start = 0
            if number:
                for offset, line in enumerate(lines, 1):
                    if "def %s(" % number in line:
                        start = offset
                        break
                else:
                    number = ""
            cited = int(digits)
            if cited > len(lines):
                add(text.citation_past_end(path_text, cited, len(lines)))
            elif number and start and start != cited:
                add(text.citation_symbol_moved(path_text, cited, number, start))
    return problems


def render_citation_note(state: Dict[str, Any]) -> str:
    """Reports the state's own file-and-line claims that the code contradicts.

    Round 9's finding: the facts are the largest thing in the state block and the
    only part of it that makes checkable claims about the code, and every one of
    those claims was true only because the round that wrote it had just looked.
    A fact pointing at the wrong line teaches the model to trust a pointer that
    was never checked, which is worse than having no pointer at all.
    """
    text = _prompts()
    if text is None:
        return ""
    problems = _citation_problems(state)
    if not problems:
        return ""
    return text.note_citation(problems)


def render_plan_note(scores: List[Any]) -> str:
    """Reports a path the previous round's plan named that the loop cannot see.

    The one observation of the code this module makes, and the only one the plan
    field is fit for: REASONING_PROTOCOL requires the plan to name the file or
    function to be changed, every round writes one, and until this nothing read
    it. A plan naming a file that is not where the loop can see it means the round
    did something other than what it said, and if that round also claimed every
    requirement was fixed, the claim and the tree disagree.

    Plans that say they are creating the path are skipped - a file that does not
    exist yet is the expected state there - and every failure of the probe means
    no note rather than no notes, because this is the first thing in the module
    that touches the filesystem.
    """
    if not scores:
        return ""
    last = scores[-1]
    missing = _missing_paths(_val(last, "plan"))
    if not missing:
        return ""
    text = _prompts()
    if text is None:
        return ""
    return text.note_plan(text.note_name_list_plain(missing, MAX_NAMES_IN_NOTE),
                          bool(_val(last, "all_pass", False)))


def _name_list(names: List[str], limit: int = MAX_NAMES_IN_NOTE) -> str:
    """Up to ``limit`` quoted names, saying so when the list was cut.

    A truncated list that reads as a complete one is its own small lie, and this
    note is read by a model deciding which requirement to look at. The wording is
    the prompt module's; this stays as the name it has always had because the
    contradiction note reaches for it by that name.
    """
    text = _prompts()
    if text is None:
        return ""
    return text.note_name_list(names, limit)


def _contradictions(entry: Any) -> List[str]:
    """Ways one score entry disagrees with itself. Empty when it is consistent.

    Three claims live in an entry - the requirements map, ``pass_count`` and
    ``all_pass`` - and the two protected brakes read them separately:
    ``check_exit`` takes ``all_pass`` on its own to decide whether the run looks
    finished and ``pass_count`` on its own to compute its plateau rule, while
    ``check_deadlock`` reads the map. Nothing in the package cross-checks them,
    and no validator can: each field is individually well typed, so a pydantic
    model accepts an entry whose three claims cannot all be true.

    An empty map is consistent with both scalars by construction, which is what
    keeps this silent on the synthetic entries the mirror tables are built from.
    """
    reqs = _val(entry, "requirements")
    if not isinstance(reqs, dict) or not reqs:
        return []
    text = _prompts()
    if text is None:
        return []
    found: List[str] = []
    non_bool = sorted(str(k) for k, v in reqs.items() if not isinstance(v, bool))
    if non_bool:
        found.append(text.contradiction_non_bool(
            _name_list(non_bool), len(non_bool) > 1))
    actual = sum(1 for v in reqs.values() if v is True)
    claimed = _val(entry, "pass_count")
    if isinstance(claimed, int) and not isinstance(claimed, bool) and claimed != actual:
        found.append(text.contradiction_pass_count(claimed, actual))
    stated = _val(entry, "all_pass")
    if isinstance(stated, bool):
        failing = sorted(str(k) for k, v in reqs.items() if v is False)
        if stated and failing:
            found.append(text.contradiction_all_pass_true(
                _name_list(failing), len(failing) > 1))
        elif not stated and actual == len(reqs):
            found.append(text.contradiction_all_pass_false(len(reqs)))
    return found


def contradiction_window(probe: Any = None) -> int:
    """How many recent entries the self-contradiction check reads.

    The window is the one ``check_exit`` takes its plateau rule from, so an entry
    outside it cannot change a brake decision and a contradiction in it is history
    rather than a hazard. That is a claim about the protected code, so when the
    code can be loaded the window is derived from it - asking ``check_exit`` for
    each verdict and keeping the window size that would reproduce it - and the
    constant is only the fallback for a probe that cannot answer. Passing a probe
    explicitly is how the derivation is tested against a controlled one.
    """
    if probe is None:
        probe = _runner()
    observed = _observed_plateau_window(probe)
    return observed if observed is not None else SCORE_CONTRADICTION_WINDOW


def render_score_contradiction_note(scores: List[Any], probe: Any = None) -> str:
    """Reports score entries that contradict themselves, over the plateau window.

    The window is derived from ``check_exit`` rather than trusted to a constant, so
    this keeps its meaning if the protected plateau clause ever moves; see
    ``contradiction_window``.

    This sits high in the block because it is a data-integrity failure rather
    than a pattern: the regression note and the repeated-cause note both read
    ``pass_count``, so every advisory conclusion drawn from a contradictory entry
    is drawn from a number the same entry refutes.
    """
    text = _prompts()
    if text is None:
        return ""
    problems: List[str] = []
    for entry in list(scores)[-contradiction_window(probe or _runner()):]:
        for problem in _contradictions(entry):
            problems.append(text.round_problem(_val(entry, "round", "?"), problem))
    if not problems:
        return ""
    return text.note_score_contradiction(problems[:MAX_NAMES_IN_NOTE])


def _runner() -> Any:
    """The runner module, imported lazily, or None when it cannot be reached.

    A module-level import would be circular: ``runner`` imports ``prompts`` and
    ``prompts`` imports this module. By the time any note renders, both are fully
    imported, so importing the runner inside a function body is safe and costs
    nothing at import time. Every failure returns None, because an aid that can
    raise while building the round prompt would break the loop it exists to help.
    """
    try:
        from jinx import runner
        return runner
    except Exception:
        return None


def _prompts() -> Any:
    """The prompt module, imported lazily, or None when it cannot be reached.

    Same shape and the same reason as ``_runner``: ``prompts`` imports this module
    at import time, so importing it back at module level would be circular. Every
    string the model reads is owned by that module - this one decides which note
    fires and with which numbers, and the wording is assembled there - so a note
    whose text cannot be reached returns '' rather than inventing English here.
    """
    try:
        from jinx import prompts
        return prompts
    except Exception:
        return None


def _window_blocked(counts: List[int], window: int) -> bool:
    """The plateau rule for one window size, over bare pass counts.

    The smallest reimplementation of ``check_exit``'s plateau clause that can be
    evaluated at any window, which is what makes it possible to ask the protected
    code which window it uses instead of trusting a constant in this file.
    """
    if len(counts) < window:
        return False
    return bool(counts[:-window]) and max(counts[-window:]) > max(counts[:-window])


def _observed_plateau_window(probe: Any) -> Optional[int]:
    """The plateau window the protected check_exit implements, or None.

    Each case in ``PLATEAU_WINDOW_CASES`` is built so that the protected verdict
    rules out every window size but one. Asking the protected function for the
    verdict, then keeping the window sizes that agree with it, derives the window
    from behaviour: if the protected window ever moves, this follows it, and
    nothing in this file has to be edited to keep the note's bound correct.

    None means the cases did not pin a single size, which is the case where this
    must not be trusted to claim anything.
    """
    if probe is None:
        return None
    possible: Optional[Set[int]] = None
    for counts, rounds, minimum in PLATEAU_WINDOW_CASES:
        scores = [{"round": number, "requirements": {"a": True}, "all_pass": True,
                   "pass_count": count} for number, count in zip(rounds, counts)]
        try:
            can_exit = bool(probe.check_exit(scores, minimum, rounds[-1]))
        except Exception:
            return None
        agrees = {size for size in range(1, len(counts) + 1)
                  if _window_blocked(counts, size) != can_exit}
        possible = agrees if possible is None else (possible & agrees)
    if not possible or len(possible) != 1:
        return None
    return possible.pop()


def _exit_probe() -> List[Tuple[List[Dict[str, Any]], int, int]]:
    """Score histories that separate the three clauses of check_exit.

    The table carries no expected answers: it exists to make the protected rule
    and the advisory mirror disagree if they ever do, not to re-state what either
    one is supposed to decide. The histories span the min_rounds floor, the
    two-round evidence requirement, the plateau refusal while still improving, the
    plateau that is allowed to exit, and a latest round that is not all_pass.
    """
    def one(passed: bool, count: int, rnd: int) -> Dict[str, Any]:
        return {"round": rnd, "all_pass": passed, "pass_count": count,
                "requirements": {"a": passed}}

    def many(rounds: Any, counts: Any) -> List[Dict[str, Any]]:
        return [{"round": r, "all_pass": True, "pass_count": c,
                 "requirements": {"a": True}} for r, c in zip(rounds, counts)]

    return [
        ([], 10, 2),
        ([one(True, 1, 1)], 10, 2),
        ([one(True, 1, 1), one(True, 1, 2)], 2, 2),
        (many((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)), 2, 5),
        (many((1, 2, 3, 4), (3, 3, 3, 3)), 2, 4),
        ([one(False, 0, 2)], 2, 2),
    ]


def _similarity_probe() -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Approach pairs that separate every branch of the similarity rule."""
    def g(nodes: Any, edges: Any = ()) -> Dict[str, Any]:
        return {"approach_graph": {
            "nodes": [{"id": n} for n in nodes],
            "edges": [{"source": s, "relation": "r", "target": t} for s, t in edges],
        }}

    return [
        (g(["a"], [("a", "b")]), g(["a", "b"], [("a", "b")])),
        (g(["a"]), g(["z"])),
        (g(["a", "b", "c"]), g(["a", "b", "d"])),
        (g(["a", "b", "c", "d"]), g(["x", "y", "z", "w"])),
        ({"approach": "same name"}, {"approach": "same name"}),
        ({"approach": "one"}, {"approach": "two"}),
    ]


def _compare_exit(check_exit: Any, scores: List[Any], min_rounds: int, rnd: int) -> bool:
    """True when the protected exit rule and the advisory mirror still agree.

    Any failure to compare - a raised exception, a changed signature, a return
    value that is not a verdict at all - counts as disagreement, because the
    alternative is to keep reporting an exit blocker computed from a rule nobody
    can confirm.
    """
    try:
        allowed = bool(check_exit(scores, min_rounds, rnd))
        mirrored = not bool(exit_blockers(scores, min_rounds, rnd))
    except Exception:
        return False
    return allowed is mirrored


def _compare_similarity(similar: Any, left: Any, right: Any) -> bool:
    """True when both similarity implementations return the same verdict."""
    try:
        return bool(similar(left, right)) is approaches_similar(left, right)
    except Exception:
        return False


def _table_status(runner_mod: Any) -> Dict[str, bool]:
    """Mirror agreement over the fixed probe tables, which cover every branch."""
    status = {"exit": True, "similarity": True}
    check_exit = getattr(runner_mod, "check_exit", None)
    if not callable(check_exit):
        status["exit"] = False
    else:
        for probe, p_min, p_rnd in _exit_probe():
            if not _compare_exit(check_exit, probe, p_min, p_rnd):
                status["exit"] = False
                break
    similar = getattr(runner_mod, "_are_approaches_similar", None)
    if not callable(similar):
        status["similarity"] = False
    else:
        for left, right in _similarity_probe():
            if not _compare_similarity(similar, left, right):
                status["similarity"] = False
                break
    return status


def _history_status(
    runner_mod: Any, scores: List[Any], min_rounds: int, rnd: int
) -> Dict[str, bool]:
    """Mirror agreement on the score history that actually exists.

    The tables cover the branches of the rules; they cannot cover the region a
    real run happens to be sitting in, because a table is chosen in advance and a
    run is not. Comparing the mirrors on the history costs one exit call and a
    handful of pairs, and a disagreement found here is a real one - the advisory
    would have reported a verdict about this run's own data from a rule that no
    longer matches the brake.

    Bounded on purpose: the widest few requirements, and only the most recent
    entries of each, since that is where the abort would be counting clusters.
    """
    status = {"exit": True, "similarity": True}
    check_exit = getattr(runner_mod, "check_exit", None)
    if not callable(check_exit):
        status["exit"] = False
    elif not _compare_exit(check_exit, scores, min_rounds, rnd):
        status["exit"] = False

    similar = getattr(runner_mod, "_are_approaches_similar", None)
    if not callable(similar):
        status["similarity"] = False
        return status
    widest = sorted(
        _failing_by_requirement(scores).items(), key=lambda kv: (-len(kv[1]), kv[0])
    )[:HISTORY_REQUIREMENTS]
    for _req, entries in widest:
        recent = entries[-HISTORY_ENTRIES_PER_REQUIREMENT:]
        for index, left in enumerate(recent):
            for right in recent[index + 1:]:
                if not _compare_similarity(similar, left, right):
                    status["similarity"] = False
                    return status
    return status


def _mirror_status(scores: List[Any], min_rounds: int, rnd: int) -> Dict[str, bool]:
    """Whether each advisory mirror still agrees with the brake logic it mirrors.

    A mirror is only as good as the last time somebody compared it, and the only
    place a comparison can live is here: the test tree is restored to the baseline
    before the self-patch gate verifies, so a test added there is deleted, and an
    import-time assertion in this module would be a side effect in production.
    A drift here is not a crash and not a wrong note - the dependent notes are
    withheld and the model is told why, because an advisory computed from a rule
    that has changed underneath it is worse than no advisory.
    """
    runner_mod = _runner()
    if runner_mod is None:
        return {"exit": True, "similarity": True}
    table = _table_status(runner_mod)
    history = _history_status(runner_mod, scores or [], min_rounds, rnd)
    return {key: table[key] and history[key] for key in table}


def render_mirror_note(status: Dict[str, bool]) -> str:
    """Says which advisory notes are being withheld, and why."""
    text = _prompts()
    if text is None:
        return ""
    lost = [key for key in ("exit", "similarity") if not status.get(key, True)]
    if not lost:
        return ""
    return text.note_mirror_drift(lost)


def render_notes(
    state_dump: str, header: str, min_rounds: int = 0, rnd: int = 0,
    max_chars: int = MAX_NOTE_CHARS,
) -> str:
    """The bounded reasoning-note block, or '' when there is nothing to say.

    ``header`` is passed in rather than defined here, and so is every string the
    model reads: the prompt module is the single home for model-facing prose.
    What stays here is the decision of which notes apply and in what order -
    priority is a judgement about what the model should read first, and it belongs
    with the detection that produces each note. The joining and the character
    budget are the prompt module's, so a note can never be handed over cut off
    mid-sentence, and an over-budget one is dropped rather than truncated.
    """
    state = state_from_dump(state_dump)
    scores = state.get("scores")
    if not isinstance(scores, list) or not scores:
        return ""
    text = _prompts()
    if text is None:
        return ""
    status = _mirror_status(scores, min_rounds, rnd)
    parts = (
        render_mirror_note(status),
        render_exit_note(scores, state, min_rounds, rnd) if status["exit"] else "",
        render_confirm_note(scores),
        render_score_contradiction_note(scores),
        render_citation_note(state),
        render_deadlock_risk_note(scores) if status["similarity"] else "",
        render_repeat_note(scores) if status["similarity"] else "",
        render_regression_note(scores),
        render_plan_note(scores),
        render_attribution_note(scores) if status["similarity"] else "",
        render_cause_note(scores),
        render_requirement_note(scores),
        render_calibration_note(scores),
    )
    return text.assemble_notes(header, list(parts), max_chars)
