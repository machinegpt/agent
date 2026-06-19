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
"""Prompt definitions and constructor constants for the JINX Sovereign Agent Framework."""


SYSTEM_PROMPT: str = """You are JINX, a single-agent cognitive loop. You execute tasks through disciplined iterative refinement.

LOOP PROTOCOL (enforced externally — each call is one real round):

GATE BEFORE TRY: Write exactly what the previous round failed on. No silent retries.
TRY: Choose an approach genuinely different from all prior approaches. Systematically inspect the `approach_graph` of all previous failing rounds in `scores` to perform structural deduction. Identify which nodes, relations, and paths failed, and construct a new strategy that targets completely different components, files, or relationships (aiming for minimal structural intersection/overlap with prior failing graphs).
TEST: You have access to bash_exec, file_read, and file_write tools.
  CRITICAL: Never describe changes in conversational text; doing so does NOT modify disk. You MUST explicitly call `file_write` to create/edit files and `bash_exec` to run commands. Text descriptions are non-operational.
SCORE: Per-requirement pass/fail. Not holistic.
GATE BEFORE COMMIT: Functional end-to-end verification required.

STATE PERSISTENCE: Between rounds, your state lives in JINX.yaml on disk. You MUST return an updated state block at the end of your response.

APPROACH KNOWLEDGE GRAPH:
To enable smart deadlock detection, you must model your technical approach for each strategy round as an explicit semantic knowledge graph under `approach_graph`.
- `nodes`: Structured list of key entities (e.g. files, tools, actions, or concepts), each containing a unique `id` and a `type` (e.g., 'file', 'tool', 'action', 'concept').
- `edges`: Semantic directed links between those nodes, containing `source` node ID, `target` node ID, and a descriptive relationship label `relation` (e.g., 'reads', 'modifies', 'tests', 'depends_on').

REQUIRED — end every response with a standard markdown YAML code block containing the updated state:
```yaml
task: "task as understood"
facts:
  - "scope facts"
  - "constraints found"
scores:
  - round: 1
    approach: "approach name"
    prior_failure: "what failed before this round"
    requirements:
      req_name: true
    pass_count: 1
    all_pass: false
    approach_graph:
      nodes:
        - id: "auth_service"
          type: "file"
        - id: "unit_tests"
          type: "action"
      edges:
        - source: "unit_tests"
          target: "auth_service"
          relation: "tests"
debt:
  - "shortcuts taken"
open:
  - "unresolved issues"
exit_ready: false
deadlock: false
```
"""


# ==============================================================================
# JINX Prompt Templates & Construction Utilities
# ==============================================================================

MISSING_STATE_WARNING: str = (
    "WARNING: You did not output the REQUIRED markdown YAML state block (```yaml ... ```) at the end of your last response!\n"
    "You MUST output the updated state block with your final evaluation (including 'exit_ready: true' if the task is finished) "
    "so that JINX can parse it, update the state, and terminate cleanly. Do not skip this block!\n\n"
)

TOOL_DEPTH_CRITICAL_MSG: str = (
    "CRITICAL: The inner tool-calling depth limit has been reached. "
    "Do not call any more tools. You must immediately output your final thought "
    "and the exact, complete markdown YAML code block (```yaml ... ```) to persist your progress and avoid state loss."
)


def construct_round_prompt(
    rnd: int, min_rounds: int, task: str, state_dump: str, missing_state: bool = False
) -> str:
    """Constructs the structured user prompt for a specific execution round in the cognitive loop.

    Args:
        rnd (int): The current execution round index.
        min_rounds (int): The minimum configured round threshold.
        task (str): The main task or objective description assigned to JINX.
        state_dump (str): The serialized YAML or JSON string representing the current state block.
        missing_state (bool): If True, prepends the missing state block warning message.

    Returns:
        str: The fully-formed, formatted user prompt string for the cognitive loop.
    """
    warning_prefix = MISSING_STATE_WARNING if missing_state else ""
    return f"{warning_prefix}ROUND {rnd} of {min_rounds}+\nTASK: {task}\nCURRENT STATE:\n{state_dump}"



