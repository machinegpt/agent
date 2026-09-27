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
"""State management and file-system serialization layer for JINX."""

import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("jinx.state")

AGENT_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Hard ceiling on how many prose facts are carried in the state block. Facts are
# re-sent every round, so an uncapped list is the dominant per-round cost once a
# task runs long. Oldest entries are dropped first.
FACTS_CAP: int = int(os.environ.get("JINX_FACTS_CAP", "60"))


def _safe_approach_text(value: Any, default: str = "unspecified") -> str:
    """Coerce LLM-produced values to a short, safe string for state summaries."""
    if value is None:
        return default
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if not text:
        return default
    return text[:80]


def _resolve_jinx_path() -> Path:
    """Resolves the JINX.yaml path dynamically."""
    env_path = os.environ.get("JINX_PATH")
    if env_path:
        return Path(env_path).resolve()

    dev_jinx_path = AGENT_DIR / "JINX.yaml"
    if dev_jinx_path.exists() and dev_jinx_path.is_file():
        return dev_jinx_path

    curr = Path.cwd().resolve()
    for parent in [curr] + list(curr.parents):
        candidate = parent / ".agent" / "JINX.yaml"
        if candidate.exists() and candidate.is_file():
            return candidate

    return Path.cwd() / ".agent" / "JINX.yaml"


# Module-level constant — replaces the fragile __getattr__ pattern.
JINX_PATH: Path = _resolve_jinx_path()


def atomic_write_yaml(path: Path, data: Any, width: int = sys.maxsize) -> None:
    """Atomically writes data to a YAML file via a temporary staging file.

    This is the single source of truth for atomic YAML writes. Both
    ``StateManager.persist_state`` and runner's ``Yaml.safe_atomic_write``
    delegate here to avoid duplicating the temp-file-replace pattern.

    The write goes to ``<path>.tmp`` and is then moved into place with
    ``Path.replace``, so a reader never observes a half-written file and a crash
    mid-write cannot corrupt the previous contents.
    """
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import io
        buf = io.StringIO()
        yaml.dump(
            data, buf, allow_unicode=True,
            default_flow_style=False, sort_keys=False, width=width,
        )
        clean_yaml = buf.getvalue()
        if clean_yaml and not clean_yaml.endswith('\n'):
            clean_yaml += '\n'
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(clean_yaml)
        temp_path.replace(path)
    except Exception as e:
        logger.error("Atomic write failed on %s: %s", path, e, exc_info=True)
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError(f"Atomic write failure on {path.name}: {e}") from e


class GraphNode(BaseModel):
    """A node in the strategy approach graph."""
    id: str
    type: str


class GraphEdge(BaseModel):
    """A semantic relationship edge in the strategy approach graph."""
    source: str
    target: str
    relation: str


class ApproachGraph(BaseModel):
    """Knowledge graph representing the agent's strategy approach."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class ScoreEntry(BaseModel):
    """Evaluation metrics and requirements score entry for a single strategy round.

    Accepts two formats:
    - Full:    {round, approach, requirements: {name: bool}, pass_count, all_pass}
    - Simplified: {round, verdict: "pass"|"fail", detail: <string>}

    The simplified format is auto-normalized to the full format on load.
    """
    round: int = 0
    approach: str = "unspecified"
    prior_failure: Optional[str] = None
    requirements: Dict[str, bool] = Field(default_factory=dict)
    pass_count: int = 0
    all_pass: bool = False
    approach_graph: Optional[ApproachGraph] = None
    # Simplified format fields (optional, auto-converted)
    verdict: Optional[str] = None
    detail: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Normalize simplified verdict format to full format."""
        if self.verdict is not None and not self.requirements:
            is_pass = str(self.verdict).lower().strip() in ("pass", "passed", "ok", "true", "1")
            self.all_pass = is_pass
            self.pass_count = 1 if is_pass else 0
            self.requirements = {"task_complete": is_pass}
            if self.approach == "unspecified":
                self.approach = _safe_approach_text(self.detail)


class StateBlock(BaseModel):
    """The structured state data preserved across agent cognitive rounds."""
    task: Optional[str] = None
    facts: List[str] = Field(default_factory=list)
    scores: List[ScoreEntry] = Field(default_factory=list)
    debt: List[str] = Field(default_factory=list)
    open: List[str] = Field(default_factory=list)
    # Durable, cross-run rules. Unlike facts/debt/open these are ADDITIVE and are
    # written to a separate ledger, so a new session does not erase them.
    lessons: List[Any] = Field(default_factory=list)
    exit_ready: bool = False
    deadlock: bool = False


class StateManager:
    """State management service with atomic disk persistence."""

    @classmethod
    def load_state(cls) -> Dict[str, Any]:
        """Loads and parses the master JINX.yaml configuration state."""
        jinx_path = _resolve_jinx_path()
        if not jinx_path.exists():
            return {}
        try:
            with open(jinx_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except (yaml.YAMLError, OSError) as e:
            logger.error("Failed to read JINX.yaml at %s: %s", jinx_path, e)
            return {}

    @classmethod
    def persist_state(cls, data: Dict[str, Any]) -> None:
        """Persists the master state to JINX.yaml atomically."""
        atomic_write_yaml(_resolve_jinx_path(), data)


def _normalize_score_entry(entry: Any) -> Dict[str, Any]:
    """Normalizes a single score entry to the full ScoreEntry format.

    Handles both:
    - Full format:    {round, approach, requirements: {name: bool}, pass_count, all_pass}
    - Simplified:     {round, verdict: "pass"|"fail", detail: <string>}
    """
    if not isinstance(entry, dict):
        return entry

    # Already in full format with requirements
    if entry.get("requirements") and isinstance(entry["requirements"], dict):
        return entry

    # Simplified verdict format
    verdict = entry.get("verdict")
    if verdict is not None:
        is_pass = str(verdict).lower().strip() in ("pass", "passed", "ok", "true", "1")
        detail_value = entry.get("detail")
        approach_value = entry.get("approach")
        normalized = {
            "round": entry.get("round", 0),
            "approach": _safe_approach_text(detail_value if detail_value is not None else approach_value),
            "requirements": {"task_complete": is_pass},
            "pass_count": 1 if is_pass else 0,
            "all_pass": is_pass,
        }
        if entry.get("prior_failure"):
            normalized["prior_failure"] = entry["prior_failure"]
        if entry.get("approach_graph"):
            normalized["approach_graph"] = entry["approach_graph"]
        return normalized

    return entry


def _normalize_state_update(update: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes the entire state update block before validation.

    Ensures all score entries are in the full ScoreEntry format so
    pydantic model_validate does not reject them.

    None-valued fields are ignored intentionally so partial updates can
    preserve existing state while still allowing explicit boolean flags like
    ``deadlock`` or ``exit_ready`` to be applied.
    """
    clean_update = {k: v for k, v in update.items() if v is not None}
    if "scores" in clean_update and isinstance(clean_update["scores"], list):
        clean_update["scores"] = [_normalize_score_entry(e) for e in clean_update["scores"]]
    return clean_update


def merge_scores(
    existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge score entries by round number instead of replacing the whole list.

    The model is asked to re-send its full score history, but it is no longer
    required to: any entry whose ``round`` already exists replaces that stored
    entry, and rounds missing from ``incoming`` are preserved. That makes a
    delta-only reply safe and removes the silent-data-loss mode where omitting a
    round permanently deleted it from disk.

    Entries lacking an integer ``round`` are keyed positionally so they survive
    round-trips without colliding with numbered entries. Unnumbered entries are
    emitted BEFORE the numbered history: ``check_exit`` reads ``scores[-1]`` as the
    current round, so a legacy entry sitting last would be mistaken for it.
    """
    merged: Dict[Any, Dict[str, Any]] = {}
    anon = 0
    for entry in list(existing or []) + list(incoming or []):
        if not isinstance(entry, dict):
            continue
        rnd = entry.get("round")
        if isinstance(rnd, int) and not isinstance(rnd, bool):
            key: Any = rnd
        else:
            key = ("anon", anon)
            anon += 1
        merged[key] = entry

    numbered = sorted((k, v) for k, v in merged.items() if isinstance(k, int))
    unnumbered = [v for k, v in merged.items() if not isinstance(k, int)]
    return unnumbered + [v for _, v in numbered]


def normalize_text_list(items: List[str], cap: int = 0) -> List[str]:
    """Deduplicate a prose list, preserving order, then apply an optional cap.

    ``facts``, ``debt`` and ``open`` are curated working memory: the model sends
    the list it wants to keep, so they are replaced rather than accumulated —
    otherwise a wrong belief could never be retracted. What this fixes is the
    growth problem, not the ownership problem: near-duplicate entries collapse to
    one, and ``facts`` (re-sent every round, so the dominant per-round cost in a
    long task) is capped, oldest dropped first.

    Duplicate detection ignores case, punctuation and whitespace, so
    "Doesn't work." and "doesnt work" collapse into a single entry.
    """
    out: List[str] = []
    seen = set()
    for raw in items or []:
        if not isinstance(raw, str):
            raw = str(raw)
        text = raw.strip()
        if not text:
            continue
        norm = _normalize_fact(text)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    if cap and len(out) > cap:
        logger.debug("text list cap %d dropped %d oldest entries", cap, len(out) - cap)
        out = out[-cap:]
    return out


def _normalize_fact(text: str) -> str:
    """Normalizes a fact for near-duplicate detection.

    Case, punctuation and whitespace are ignored, so "Doesn't work." and
    "doesnt work" collapse to one entry.
    """
    lowered = text.lower()
    stripped = "".join(ch for ch in lowered if ch.isalnum() or ch.isspace())
    return " ".join(stripped.split())


def read_jinx() -> Dict[str, Any]:
    """Reads and parses the JINX state manifest file."""
    return StateManager.load_state()


def write_jinx(data: Dict[str, Any]) -> None:
    """Serializes the configuration dictionary to JINX.yaml on disk.

    Raises OSError if persistence fails.
    """
    StateManager.persist_state(data)


def merge_state(
    jinx: Dict[str, Any], update: Dict[str, Any],
    diagnostics: Optional[List[str]] = None,
    outcome: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merges a parsed update block back into the JINX manifest state.

    Automatically normalizes simplified score formats (verdict/detail)
    to the full ScoreEntry format before validation.

    Scores are merged by round number and prose lists are deduplicated, so a
    reply that carries only the current round cannot destroy history. When
    validation fails the previous state is kept AND the reason is appended to
    ``diagnostics`` so the caller can tell the model why its block was dropped —
    previously the rejection was logged and then invisible.

    ``outcome``, when supplied, receives ``{"applied": bool}``. Callers that act
    on ``exit_ready``/``deadlock`` MUST consult it: those flags are only
    trustworthy once the whole block has passed validation, and honouring them
    from a rejected block would let a malformed response terminate the loop or
    claim success on stale state.
    """
    if "state" in update and isinstance(update["state"], dict):
        # Merge protocol section into jinx top-level so _resolve_min_rounds can read it
        if "protocol" in update and isinstance(update["protocol"], dict):
            jinx.setdefault("protocol", {}).update(update["protocol"])
        update = update["state"]

    # Normalize before validation — handles verdict/detail -> all_pass/requirements
    update = _normalize_state_update(update)

    try:
        validated_block = StateBlock.model_validate(update)
        validated_dict = validated_block.model_dump(exclude_none=True)
    except Exception as e:
        logger.error("State validation failed: %s. Rejecting update.", e)
        if diagnostics is not None:
            diagnostics.append(
                "Your previous state block was REJECTED and discarded; the state on "
                "disk is unchanged. Reason: %s: %s. Re-send a corrected block. Common "
                "causes: an unquoted ':' or '#' inside a scalar value, a tab used for "
                "indentation, or a key nested one level too deep. Because the block "
                "was rejected, your exit_ready/deadlock flags were NOT honoured."
                % (type(e).__name__, str(e).splitlines()[0][:200])
            )
        if outcome is not None:
            outcome["applied"] = False
            outcome["error"] = str(e)
        return jinx

    if outcome is not None:
        outcome["applied"] = True

    s: Dict[str, Any] = jinx.setdefault("state", {})

    # Lessons are durable and cross-run, so they are deliberately NOT written
    # into the state block: ``_init_new_session`` resets this dict when a new
    # task starts, and re-sending them here would cost tokens every round for
    # no reason. They are handed to the caller to persist in the ledger.
    if "lessons" in update:
        if outcome is not None:
            outcome["lessons"] = update.get("lessons")
        else:
            update = {k: v for k, v in update.items() if k != "lessons"}

    for key in ("task",):
        if key in update and key in validated_dict:
            s[key] = validated_dict[key]

    if "scores" in update and "scores" in validated_dict:
        merged = merge_scores(s.get("scores") or [], validated_dict["scores"])
        s["scores"] = merged
        if diagnostics is not None:
            stored = len(s.get("scores") or [])
            sent = len(validated_dict["scores"] or [])
            if stored > sent:
                diagnostics.append(
                    "State accepted. Score history merged by round: %d entr%s on "
                    "disk, %d sent this round — %d preserved from earlier rounds. "
                    "You may send only the current round's entry from now on."
                    % (stored, "y" if stored == 1 else "ies", sent, stored - sent)
                )

    # facts/debt/open are curated working memory: the model's list wins, but
    # duplicates collapse and facts are capped so the per-round cost stays flat.
    for key, cap in (("facts", FACTS_CAP), ("debt", 0), ("open", 0)):
        if key in update and key in validated_dict:
            s[key] = normalize_text_list(validated_dict[key] or [], cap=cap)

    if "scores" in s and isinstance(s["scores"], list) and len(s["scores"]) > 5:
        for entry in s["scores"][:-5]:
            if isinstance(entry, dict):
                entry.pop("prior_failure", None)

    if "exit_ready" in update:
        s["exit_ready"] = validated_block.exit_ready
    if "deadlock" in update:
        s["deadlock"] = validated_block.deadlock
    return jinx
