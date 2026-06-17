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
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

# Setup a dedicated stderr logger for internal system logs
logger = logging.getLogger("jinx.state")

# Resolve the base directory where the agent is installed (e.g., .agent/)
AGENT_DIR: Path = Path(__file__).resolve().parent.parent.parent
JINX_PATH: Path = AGENT_DIR / "JINX.yaml"


class ScoreEntry(BaseModel):
    """Evaluation metrics and requirements score entry for a single strategy round."""

    round: int
    approach: str
    prior_failure: Optional[str] = None
    requirements: Dict[str, bool] = Field(default_factory=dict)
    pass_count: int = 0
    all_pass: bool = False


class StateBlock(BaseModel):
    """The structured state data preserved across agent cognitive rounds."""

    task: Optional[str] = None
    facts: List[str] = Field(default_factory=list)
    scores: List[ScoreEntry] = Field(default_factory=list)
    debt: List[str] = Field(default_factory=list)
    open: List[str] = Field(default_factory=list)
    exit_ready: bool = False
    deadlock: bool = False


def read_jinx() -> Dict[str, Any]:
    """Reads and parses the JINX state manifest file (JINX.yaml).

    Returns:
        Dict[str, Any]: The parsed configuration dictionary, or an empty
            dictionary if the file does not exist or fails to parse.
    """
    if not JINX_PATH.exists():
        logger.debug("JINX state file not found at %s. Returning empty state.", JINX_PATH)
        return {}
    try:
        with open(JINX_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError) as e:
        logger.error("Failed to read or parse JINX.yaml at %s: %s", JINX_PATH, e)
        return {}


def write_jinx(data: Dict[str, Any]) -> None:
    """Serializes the configuration dictionary to JINX.yaml on disk.

    Args:
        data (Dict[str, Any]): The state manifest dictionary to serialize.
    """
    try:
        # Ensure parent directory exists before writing
        JINX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JINX_PATH, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
                encoding="utf-8"
            )
    except OSError as e:
        logger.error("Failed to write JINX state serialization to %s: %s", JINX_PATH, e)


def merge_state(jinx: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merges a parsed update block back into the JINX manifest state.

    Args:
        jinx (Dict[str, Any]): The existing master manifest configuration.
        update (Dict[str, Any]): The new incremental state update from the agent.

    Returns:
        Dict[str, Any]: The updated master manifest configuration block.
    """
    s: Dict[str, Any] = jinx.setdefault("state", {})
    for key in ("task", "facts", "scores", "debt", "open", "exit_ready", "deadlock"):
        if key in update:
            s[key] = update[key]
    return jinx
