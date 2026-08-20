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

    Post-processes YAML to remove blank lines for compact output.
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
        raw = buf.getvalue()
        lines = raw.splitlines()
        cleaned: List[str] = []
        for line in lines:
            if line.strip() == "":
                cleaned.append(line)
                continue
            cleaned.append(line)
        clean_yaml = '\n'.join(cleaned)
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


def read_jinx() -> Dict[str, Any]:
    """Reads and parses the JINX state manifest file."""
    return StateManager.load_state()


def write_jinx(data: Dict[str, Any]) -> None:
    """Serializes the configuration dictionary to JINX.yaml on disk.

    Raises OSError if persistence fails.
    """
    StateManager.persist_state(data)


def merge_state(jinx: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merges a parsed update block back into the JINX manifest state.

    Automatically normalizes simplified score formats (verdict/detail)
    to the full ScoreEntry format before validation.
    """
    # If the update contains a nested 'state' key (from the full YAML block
    # including id/protocol), extract just the state fields for validation.
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
        return jinx

    s: Dict[str, Any] = jinx.setdefault("state", {})
    for key in ("task", "facts", "scores", "debt", "open"):
        if key in update and key in validated_dict:
            s[key] = validated_dict[key]

    if "scores" in s and isinstance(s["scores"], list) and len(s["scores"]) > 5:
        for entry in s["scores"][:-5]:
            entry.pop("prior_failure", None)

    if "exit_ready" in update:
        s["exit_ready"] = validated_block.exit_ready
    if "deadlock" in update:
        s["deadlock"] = validated_block.deadlock
    return jinx
