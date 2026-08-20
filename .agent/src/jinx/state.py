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
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("jinx.state")

AGENT_DIR: Path = Path(__file__).resolve().parent.parent.parent


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


def __getattr__(name: str) -> Any:
    """Handles dynamic lookups for module-level attributes."""
    if name == "JINX_PATH":
        return _resolve_jinx_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    """Evaluation metrics and requirements score entry for a single strategy round."""
    round: int = 0
    approach: str = "unspecified"
    prior_failure: Optional[str] = None
    requirements: Dict[str, bool] = Field(default_factory=dict)
    pass_count: int = 0
    all_pass: bool = False
    approach_graph: Optional[ApproachGraph] = None


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
            raise OSError(f"State read failure on {jinx_path.name}: {e}") from e

    @classmethod
    def persist_state(cls, data: Dict[str, Any]) -> None:
        """Persists the master state to JINX.yaml atomically."""
        jinx_path = _resolve_jinx_path()
        temp_path = jinx_path.with_suffix(jinx_path.suffix + ".tmp")
        try:
            jinx_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    data, f, allow_unicode=True,
                    default_flow_style=False, sort_keys=False, width=sys.maxsize
                )
            temp_path.replace(jinx_path)
        except Exception as e:
            logger.error("Atomic persistence failed on %s: %s", jinx_path, e, exc_info=True)
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise OSError(f"State write failure on {jinx_path.name}: {e}") from e


def read_jinx() -> Dict[str, Any]:
    """Reads and parses the JINX state manifest file."""
    return StateManager.load_state()


def write_jinx(data: Dict[str, Any]) -> None:
    """Serializes the configuration dictionary to JINX.yaml on disk.

    Raises OSError if persistence fails.
    """
    StateManager.persist_state(data)


def merge_state(jinx: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merges a parsed update block back into the JINX manifest state."""
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

    s["exit_ready"] = validated_block.exit_ready
    s["deadlock"] = validated_block.deadlock
    return jinx
