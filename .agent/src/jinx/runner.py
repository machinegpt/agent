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
"""Cognitive loop orchestration, execution controller, and IDE-IPC layer for JINX."""

import json
import logging
import re
import sys
import textwrap
import atexit
import signal
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import yaml

from .prompts import SYSTEM_PROMPT, TOOL_DEPTH_CRITICAL_MSG, construct_round_prompt
from .state import merge_state, read_jinx, write_jinx
from .tools import tool_schema

logger = logging.getLogger("jinx.runner")

HARD_CAP: int = 40
TOOL_DEPTH_CAP: int = 20


class Dumper(yaml.SafeDumper):
    """Isolated PyYAML dumper class for JINX serialization."""
    pass


def str_presenter(dumper: Dumper, data: str) -> Any:
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


Dumper.add_representer(str, str_presenter)


class JinxError(Exception):
    """Base exception for all JINX Framework errors."""
    pass


class SerializationError(JinxError):
    """Raised when serialization or deserialization fails."""
    pass


class IPCError(JinxError):
    """Raised during IPC file operations or stream communication."""
    pass


class Yaml:
    """YAML serialization engine with atomic writes for JINX operations."""

    @staticmethod
    def dump_to_string(data: Any, width: int = sys.maxsize) -> str:
        """Serializes structures to YAML strings using the isolated dumper."""
        try:
            return yaml.dump(
                data, Dumper=Dumper, allow_unicode=True,
                default_flow_style=False, sort_keys=False, width=width
            )
        except Exception as e:
            raise SerializationError(f"Failed to serialize YAML string: {e}") from e

    @staticmethod
    def safe_atomic_write(path: Path, data: Any, width: int = sys.maxsize) -> None:
        """Writes data to files atomically via temporary staging files."""
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    data, f, Dumper=Dumper, allow_unicode=True,
                    default_flow_style=False, sort_keys=False, width=width
                )
            temp_path.replace(path)
        except Exception as e:
            logger.error("Atomic write failed on %s: %s", path, e, exc_info=True)
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise IPCError(f"File-IPC write failure on {path.name}: {e}") from e

    @staticmethod
    def load_from_file(path: Path) -> Any:
        """Safely loads and parses YAML structures from disk."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise SerializationError(f"Failed to load YAML file at {path}: {e}") from e


# Backwards-compatibility aliases / wrappers
# These keep older import/usage sites working when names changed.
JinxYamlDumper = Dumper
JinxEnterpriseYamlDumper = Dumper
# Historically a ValidationError symbol was exported; map it to the
# serialization-level error currently raised for YAML issues.
ValidationError = SerializationError
EnterpriseYamlEngine = Yaml

def safe_atomic_write_yaml(path: Path, data: Any, width: int = sys.maxsize) -> None:
    """Compatibility wrapper for older safe_atomic_write_yaml API."""
    return Yaml.safe_atomic_write(path, data, width)


def parse_state_block(text: str) -> Optional[Dict[str, Any]]:
    """Extracts and parses the JINX state block from markdown code fences."""
    code_block_pattern = r"[ \t]*```(?:json|yaml|yml)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
    code_matches = list(re.finditer(code_block_pattern, text, re.DOTALL))

    if code_matches:
        for match in reversed(code_matches):
            raw = textwrap.dedent(match.group(1)).strip()
            try:
                data = yaml.safe_load(raw)
                if isinstance(data, dict):
                    state_keys = {"task", "facts", "scores", "debt", "open", "exit_ready", "deadlock"}
                    if len(set(data.keys()) & state_keys) >= 2:
                        return data
            except yaml.YAMLError:
                continue

    logger.debug("No valid state block found in response.")
    return None


def _validate_tool_use_block(block: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Validate a single `tool_use` block and normalize its `input`.

    Returns (id, name, params, error_result). If validation fails, id/name/params
    are None and error_result contains a `tool_result` dict describing the error.
    """
    tool_use_id = block.get("id")
    name = block.get("name")
    params = block.get("input") if ("input" in block) else {}

    if not isinstance(tool_use_id, str) or not isinstance(name, str):
        logger.error("Malformed tool_use block: id=%r name=%r", tool_use_id, name)
        return None, None, None, {
            "type": "tool_result", "tool_use_id": tool_use_id or "",
            "content": "Error: Malformed tool_use block (missing id or name)."
        }

    if params is None:
        params = {}

    if not isinstance(params, dict):
        logger.error("Malformed tool_use block input: %r", params)
        return None, None, None, {
            "type": "tool_result", "tool_use_id": tool_use_id,
            "content": "Error: Malformed tool_use block (input must be an object)."
        }

    return tool_use_id, name, params, None


def check_exit(scores: List[Dict[str, Any]], min_rounds: int, rnd: int) -> bool:
    """Evaluates whether the cognitive loop is ready to terminate."""
    if rnd < min_rounds:
        return False
    if not scores or not scores[-1].get("all_pass") or len(scores) < 2:
        return False

    if len(scores) >= 4:
        last3_best = max(s.get("pass_count", 0) for s in scores[-3:])
        prior_history = scores[:-3]
        if prior_history:
            prior_best = max((s.get("pass_count", 0) for s in prior_history), default=0)
            if last3_best > prior_best:
                return False

    return True


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to get a value from either a dictionary or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def _are_approaches_similar(entry1: Any, entry2: Any) -> bool:
    """Calculates semantic similarity between two approach graphs or falls back to text-matching."""
    graph1 = _get_val(entry1, "approach_graph")
    graph2 = _get_val(entry2, "approach_graph")

    def extract_graph_data(g: Any) -> Optional[Dict[str, Any]]:
        if g is None:
            return None
        if hasattr(g, "model_dump"):
            return g.model_dump()
        if isinstance(g, dict):
            return g
        return None

    g1 = extract_graph_data(graph1)
    g2 = extract_graph_data(graph2)

    if not g1 or not g2:
        return _get_val(entry1, "approach", "") == _get_val(entry2, "approach", "")

    def node_ids(g: Dict[str, Any]) -> set:
        raw = g.get("nodes")
        if not isinstance(raw, list):
            return set()
        return {
            str(n.get("id", "")).strip().lower()
            for n in raw if isinstance(n, dict) and n.get("id")
        }
    def edge_keys(g: Dict[str, Any]) -> set:
        raw = g.get("edges")
        if not isinstance(raw, list):
            return set()
        return {
            (
                str(e.get("source", "")).strip().lower(),
                str(e.get("relation", "")).strip().lower(),
                str(e.get("target", "")).strip().lower(),
            )
            for e in raw
            if isinstance(e, dict) and e.get("source") and e.get("target")
        }
    nodes1, nodes2 = node_ids(g1), node_ids(g2)
    edges1, edges2 = edge_keys(g1), edge_keys(g2)
    if not nodes1 and not edges1 and not nodes2 and not edges2:
        return _get_val(entry1, "approach", "") == _get_val(entry2, "approach", "")
    node_sim = len(nodes1.intersection(nodes2)) / len(nodes1.union(nodes2)) if (nodes1 or nodes2) else None
    edge_sim = len(edges1.intersection(edges2)) / len(edges1.union(edges2)) if (edges1 or edges2) else None
    if node_sim is not None and edge_sim is not None:
        return (0.5 * node_sim + 0.5 * edge_sim) >= 0.7
    if node_sim is not None:
        return node_sim >= 0.7
    if edge_sim is not None:
        return edge_sim >= 0.7
    return False


def _select_representative(cluster: List[Any], entry: Any) -> Any:
    """Selects the first similar representative of the cluster."""
    for member in cluster:
        if _are_approaches_similar(entry, member):
            return member
    return None


def check_deadlock(scores: List[Any], min_rounds: int, rnd: int) -> bool:
    """Determines if the cognitive loop is stuck in a deadlock."""
    if rnd < min_rounds:
        return False

    failing_entries_by_req: Dict[str, List[Any]] = {}
    for entry in scores:
        for req, passed in (_get_val(entry, "requirements") or {}).items():
            if not passed:
                failing_entries_by_req.setdefault(req, []).append(entry)

    for req, entries in failing_entries_by_req.items():
        clusters: List[List[Any]] = []
        for entry in entries:
            matched_cluster = None
            for cluster in clusters:
                if _select_representative(cluster, entry) is not None:
                    matched_cluster = cluster
                    break
            if matched_cluster is not None:
                matched_cluster.append(entry)
            else:
                clusters.append([entry])

        if len(clusters) >= 3:
            logger.warning("Deadlock on '%s': %d unique strategy clusters.", req, len(clusters))
            return True

    return False


def get_tool_result_from_editor(tool_use_id: str, name: str, params: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """Dispatches tool invocation via JSON-RPC and awaits result from stdin."""
    payload = {"jinx_command": name, "tool_use_id": tool_use_id, "params": params}
    try:
        print(json.dumps(payload), flush=True)
    except OSError as e:
        logger.error("Failed to transmit tool payload: %s", e)
        return f"Error: Failed to transmit payload: {e}", False, True

    try:
        line = sys.stdin.readline()
        if not line:
            return "Error: Editor disconnected.", False, True
        response = json.loads(line)
        status = response.get("status", "")
        is_error = "error" in response or (isinstance(status, str) and "error" in status.lower())
        output = response.get("output") or response.get("content") or response.get("error") or str(response)
        was_sliced = bool(response.get("sliced") or response.get("is_sliced"))
        return str(output), was_sliced, is_error
    except (json.JSONDecodeError, OSError) as e:
        return f"Error receiving input: {e}", False, True


def request_llm_from_editor(
    system: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Delegates LLM generation to the host editor via IPC."""
    payload = {
        "jinx_command": "llm_generate",
        "params": {"system": system, "messages": messages, "tools": tools if tools is not None else tool_schema()}
    }
    try:
        print(json.dumps(payload), flush=True)
    except OSError as e:
        logger.error("Failed to transmit LLM request: %s", e)
        return []

    try:
        line = sys.stdin.readline()
        if not line:
            return []
        data = json.loads(line)
        content = data.get("content") or []
        return content if isinstance(content, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Error receiving LLM response: %s", e)
        return []


AGENT_DIR: Path = Path(__file__).resolve().parent.parent.parent
REQUEST_PATH: Path = AGENT_DIR / "jinx_request.yaml"
RESPONSE_PATH: Path = AGENT_DIR / "jinx_response.yaml"
RUN_STATE_PATH: Path = AGENT_DIR / "jinx_run_state.yaml"


def clean_up_ipc_files() -> None:
    """Removes temporary IPC communication files."""
    for p in (REQUEST_PATH, RESPONSE_PATH, RUN_STATE_PATH):
        p.unlink(missing_ok=True)


# Register cleanup handlers to ensure IPC files are removed on exit/signals.
def _signal_cleanup(signum=None, frame=None) -> None:
    try:
        logger.info("Signal %s received: cleaning up IPC files.", signum)
    except Exception:
        pass
    try:
        clean_up_ipc_files()
    except Exception:
        pass
    # On signal, exit with non-zero to indicate external termination.
    try:
        sys.exit(1)
    except SystemExit:
        raise


atexit.register(clean_up_ipc_files)
for sig in ("SIGINT", "SIGTERM", "SIGHUP"):
    try:
        signum = getattr(signal, sig)
        signal.signal(signum, _signal_cleanup)
    except (AttributeError, OSError, RuntimeError):
        # Some signals may not be available on all platforms (e.g., SIGHUP on Windows)
        continue


def _resolve_min_rounds(jinx: Dict[str, Any], min_override: Optional[int]) -> int:
    """Resolves the minimum rounds configuration from override or JINX.yaml."""
    if min_override is not None:
        return min_override
    protocol_config = jinx.get("protocol")
    if isinstance(protocol_config, dict):
        loop_config = protocol_config.get("loop")
        if isinstance(loop_config, dict):
            configured_min = loop_config.get("min")
            if isinstance(configured_min, int):
                return configured_min
    return 10


def _init_new_session(task: str, jinx: Dict[str, Any]) -> None:
    """Initializes a fresh task session in the JINX state."""
    if not isinstance(jinx.get("state"), dict):
        jinx["state"] = {}
    jinx["state"].update({
        "task": task, "facts": [], "scores": [], "debt": [],
        "open": [], "exit_ready": False, "deadlock": False
    })
    write_jinx(jinx)


def write_llm_request(
    history: List[Dict[str, Any]], rnd: int, tool_depth: int, min_rounds: int, task: str
) -> None:
    """Writes the current prompt/history state and requests LLM generation."""
    request_payload = {
        "type": "llm_generate", "system": SYSTEM_PROMPT,
        "messages": history, "tools": tool_schema()
    }
    try:
        Yaml.safe_atomic_write(REQUEST_PATH, request_payload)
    except JinxError as e:
        raise IPCError(f"Failed to write request: {e}") from e

    run_state = {
        "rnd": rnd, "tool_depth": tool_depth, "history": history,
        "waiting_for": "llm_generate", "min_rounds": min_rounds, "task": task
    }
    try:
        Yaml.safe_atomic_write(RUN_STATE_PATH, run_state)
    except JinxError as e:
        raise IPCError(f"Failed to write run state: {e}") from e

    print(f"[JINX_WAITING] Requesting LLM completion for Round {rnd}...", flush=True)


def run_file_ipc(task: Optional[str], min_override: Optional[int]) -> None:
    """Orchestrates JINX loop using a stateless File-based IPC protocol."""
    is_resuming = RUN_STATE_PATH.exists() and not task

    if not is_resuming:
        if not task:
            logger.error("Cannot start new session without a task description.")
            sys.exit(1)

        jinx = read_jinx()
        _init_new_session(task, jinx)
        min_rounds = _resolve_min_rounds(jinx, min_override)
        clean_up_ipc_files()

        state_dump = Yaml.dump_to_string(jinx["state"])
        user_msg = construct_round_prompt(rnd=1, min_rounds=min_rounds, task=task, state_dump=state_dump)
        try:
            write_llm_request([{"role": "user", "content": user_msg}], 1, 0, min_rounds, task)
        except (IPCError, OSError, JinxError) as e:
            logger.error("Failed to write initial LLM request: %s", e, exc_info=True)
            clean_up_ipc_files()
            sys.exit(1)
        return

    # Resume path
    try:
        with open(RUN_STATE_PATH, "r", encoding="utf-8") as f:
            run_state = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        logger.error("Failed to load run state: %s", e)
        clean_up_ipc_files()
        sys.exit(1)

    required_keys = ("rnd", "tool_depth", "history", "waiting_for", "min_rounds", "task")
    missing = [k for k in required_keys if k not in run_state]
    if missing:
        logger.error("Run state is incomplete. Missing keys: %s", ", ".join(missing))
        clean_up_ipc_files()
        sys.exit(1)
    rnd = run_state["rnd"]
    tool_depth = run_state["tool_depth"]
    history = run_state["history"]
    waiting_for = run_state["waiting_for"]
    min_rounds = run_state["min_rounds"]
    task = run_state["task"]

    if not RESPONSE_PATH.exists():
        logger.error("Awaiting editor response at %s", RESPONSE_PATH)
        sys.exit(1)

    try:
        with open(RESPONSE_PATH, "r", encoding="utf-8") as f:
            response_data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        logger.error("Failed to read response YAML: %s", e)
        sys.exit(1)

    RESPONSE_PATH.unlink(missing_ok=True)

    try:
        if waiting_for == "llm_generate":
            try:
                _handle_llm_response(response_data, history, rnd, tool_depth, min_rounds, task, run_state)
            except (IPCError, OSError, JinxError) as e:
                logger.error("IPC failure while handling LLM response: %s", e, exc_info=True)
                clean_up_ipc_files()
                sys.exit(1)
        elif waiting_for == "tool_calls":
            try:
                _handle_tool_response(response_data, history, rnd, tool_depth, min_rounds, task, run_state)
            except (IPCError, OSError, JinxError) as e:
                logger.error("IPC failure while handling tool response: %s", e, exc_info=True)
                clean_up_ipc_files()
                sys.exit(1)
        else:
            logger.error("Unexpected waiting_for state: '%s'", waiting_for)
            clean_up_ipc_files()
            sys.exit(1)
    except OSError as e:
        # Handle persistence failures (e.g. write_jinx -> StateManager.persist_state)
        logger.error("File-IPC persistence error while handling response: %s", e, exc_info=True)
        # Ensure temporary IPC artifacts are removed so a stale RUN_STATE_PATH
        # cannot block future runs when the response file has already been removed.
        clean_up_ipc_files()
        sys.exit(1)


def _handle_llm_response(
    response_data: Dict[str, Any], history: List[Dict[str, Any]],
    rnd: int, tool_depth: int, min_rounds: int, task: str,
    run_state: Dict[str, Any]
) -> None:
    """Handles the response from an LLM generation request."""
    raw_content = response_data.get("content") or []
    if isinstance(raw_content, str):
        content_blocks = [{"type": "text", "text": raw_content}]
    elif isinstance(raw_content, list):
        content_blocks = [
            b if isinstance(b, dict) else {"type": "text", "text": str(b)}
            for b in raw_content
        ]
    else:
        content_blocks = [{"type": "text", "text": str(raw_content)}]

    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    history.append({"role": "assistant", "content": content_blocks})

    full_text = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )

    tool_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
    valid_calls: List[Dict[str, Any]] = []
    malformed_results: List[Dict[str, Any]] = []

    for b in tool_blocks:
        tool_use_id, name, params, err = _validate_tool_use_block(b)
        if err:
            malformed_results.append(err)
            continue
        valid_calls.append({"id": tool_use_id, "name": name, "params": params})

    if malformed_results:
        history.append({"role": "user", "content": malformed_results})

    if valid_calls:
        try:
            _write_tool_request(valid_calls, history, rnd, tool_depth + 1, min_rounds, task, run_state)
        except (IPCError, OSError, JinxError) as e:
            logger.error("IPC failure while writing tool_calls request: %s", e, exc_info=True)
            clean_up_ipc_files()
            sys.exit(1)
        return

    # No tool calls — parse state block
    update = parse_state_block(full_text)
    jinx = read_jinx()
    if update:
        jinx = merge_state(jinx, update)
        write_jinx(jinx)
        scores = jinx["state"].get("scores", [])

        if update.get("exit_ready") and check_exit(scores, min_rounds, rnd):
            clean_up_ipc_files()
            print("[JINX_COMPLETE] Task resolved successfully!", flush=True)
            return

        if update.get("deadlock") or check_deadlock(scores, min_rounds, rnd):
            if not update.get("deadlock"):
                jinx["state"]["deadlock"] = True
                write_jinx(jinx)
            clean_up_ipc_files()
            print("[JINX_DEADLOCK] Loop aborted due to strategy deadlock.", flush=True)
            return

    # Transition to next round
    rnd += 1
    if rnd >= HARD_CAP:
        clean_up_ipc_files()
        logger.error("Cognitive loop exhausted HARD_CAP.")
        sys.exit(2)

    jinx = read_jinx()
    state_dump = Yaml.dump_to_string(jinx.get("state") or {})
    user_msg = construct_round_prompt(rnd=rnd, min_rounds=min_rounds, task=task, state_dump=state_dump, missing_state=not update)
    history.append({"role": "user", "content": user_msg})
    try:
        write_llm_request(history, rnd, 0, min_rounds, task)
    except (IPCError, OSError, JinxError) as e:
        logger.error("IPC failure while writing next LLM request: %s", e, exc_info=True)
        clean_up_ipc_files()
        sys.exit(1)


def _handle_tool_response(
    response_data: Dict[str, Any], history: List[Dict[str, Any]],
    rnd: int, tool_depth: int, min_rounds: int, task: str,
    run_state: Dict[str, Any]
) -> None:
    """Handles the response from tool execution."""
    results = response_data.get("results") or []
    tool_results = [
        {"type": "tool_result", "tool_use_id": r.get("tool_use_id"), "content": r.get("content") or ""}
        for r in results
    ]

    if tool_depth >= TOOL_DEPTH_CAP:
        logger.warning("Tool depth limit reached. Forcing state recovery.")
        tool_results.append({"type": "text", "text": TOOL_DEPTH_CRITICAL_MSG})
        history.append({"role": "user", "content": tool_results})
        try:
            _write_llm_request_no_tools(history, rnd, run_state)
        except (IPCError, OSError, JinxError) as e:
            logger.error("IPC failure while writing final summary request: %s", e, exc_info=True)
            clean_up_ipc_files()
            sys.exit(1)
        return

    history.append({"role": "user", "content": tool_results})
    try:
        write_llm_request(history, rnd, tool_depth, min_rounds, task)
    except (IPCError, OSError, JinxError) as e:
        logger.error("IPC failure while writing LLM request after tool response: %s", e, exc_info=True)
        clean_up_ipc_files()
        sys.exit(1)


def _write_tool_request(
    tool_calls: List[Dict[str, Any]], history: List[Dict[str, Any]],
    rnd: int, tool_depth: int, min_rounds: int, task: str,
    run_state: Dict[str, Any]
) -> None:
    """Writes a tool_calls request and updates run state."""
    request_payload = {"type": "tool_calls", "calls": tool_calls}
    try:
        Yaml.safe_atomic_write(REQUEST_PATH, request_payload)
    except JinxError as e:
        # Propagate as IPCError so callers can clean up IPC files.
        raise IPCError(f"Failed to write tool_calls request: {e}") from e

    run_state.update({"tool_depth": tool_depth, "history": history, "waiting_for": "tool_calls"})
    try:
        Yaml.safe_atomic_write(RUN_STATE_PATH, run_state)
    except JinxError as e:
        raise IPCError(f"Failed to write run state: {e}") from e

    print(f"[JINX_WAITING] Requesting tool execution for Round {rnd}...", flush=True)


def _write_llm_request_no_tools(
    history: List[Dict[str, Any]], rnd: int, run_state: Dict[str, Any]
) -> None:
    """Writes an LLM request with empty tools list (for final summary)."""
    request_payload = {
        "type": "llm_generate", "system": SYSTEM_PROMPT, "messages": history, "tools": []
    }
    try:
        Yaml.safe_atomic_write(REQUEST_PATH, request_payload)
    except JinxError as e:
        raise IPCError(f"Failed to write final summary request: {e}") from e

    run_state.update({"waiting_for": "llm_generate", "history": history})
    try:
        Yaml.safe_atomic_write(RUN_STATE_PATH, run_state)
    except JinxError as e:
        raise IPCError(f"Failed to write run state: {e}") from e

    print(f"[JINX_WAITING] Requesting final summary for Round {rnd}...", flush=True)


def run(task: Optional[str], min_override: Optional[int], ipc_mode: str = "file") -> None:
    """Orchestrates the JINX execution loop."""
    if ipc_mode == "file":
        run_file_ipc(task, min_override)
        return

    # Interactive duplex stream JSON-RPC mode
    jinx = read_jinx()
    _init_new_session(task or "", jinx)
    min_rounds = _resolve_min_rounds(jinx, min_override)

    rnd: int = 0
    last_round_missing_state: bool = False

    logger.info("Starting JINX loop (JSON-RPC). Task: '%s'. Min rounds: %d", task, min_rounds)

    while rnd < HARD_CAP:
        rnd += 1
        history: List[Dict[str, Any]] = []
        jinx = read_jinx()
        state_data = jinx.get("state") or {}
        state_dump = Yaml.dump_to_string(state_data)

        user_msg = construct_round_prompt(
            rnd=rnd, min_rounds=min_rounds, task=task,
            state_dump=state_dump, missing_state=last_round_missing_state
        )
        history.append({"role": "user", "content": user_msg})

        full_text: str = ""
        tool_depth: int = 0
        while True:
            content_blocks = request_llm_from_editor(SYSTEM_PROMPT, history)
            history.append({"role": "assistant", "content": content_blocks})

            for block in content_blocks:
                if block.get("type") == "text":
                    full_text += block.get("text", "")

            tool_results: List[Dict[str, Any]] = []
            for block in content_blocks:
                if block.get("type") == "tool_use":
                    tool_results.append(_execute_rpc_tool(block))

            if tool_results:
                tool_depth += 1
                if tool_depth >= TOOL_DEPTH_CAP:
                    logger.warning("Tool depth limit reached in RPC mode.")
                    tool_results.append({"type": "text", "text": TOOL_DEPTH_CRITICAL_MSG})
                    history.append({"role": "user", "content": tool_results})
                    content_blocks = request_llm_from_editor(SYSTEM_PROMPT, history, tools=[])
                    history.append({"role": "assistant", "content": content_blocks})
                    for block in content_blocks:
                        if block.get("type") == "text":
                            full_text += block.get("text", "")
                    break
                history.append({"role": "user", "content": tool_results})
                continue
            else:
                break

        update = parse_state_block(full_text)
        if update:
            last_round_missing_state = False
            jinx = merge_state(jinx, update)
            write_jinx(jinx)
            scores = jinx["state"].get("scores", [])

            if update.get("exit_ready") and check_exit(scores, min_rounds, rnd):
                logger.info("Execution complete in round %d.", rnd)
                break
            if update.get("deadlock") or check_deadlock(scores, min_rounds, rnd):
                logger.warning("Deadlock in round %d.", rnd)
                if not update.get("deadlock"):
                    jinx["state"]["deadlock"] = True
                    write_jinx(jinx)
                break
        else:
            last_round_missing_state = True
    else:
        logger.error("HARD_CAP (%d rounds) exhausted.", HARD_CAP)
        sys.exit(2)


def _execute_rpc_tool(block: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a single tool call in RPC mode."""
    tool_use_id, name, params, err = _validate_tool_use_block(block)
    if err:
        return err
    result_content, was_sliced, is_error = get_tool_result_from_editor(tool_use_id, name, params)
    if name == "file_read" and not is_error and not was_sliced:
        result_content = _slice_file_content(result_content, params)
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": result_content}


def _slice_file_content(result_content: str, params: Dict[str, Any]) -> str:
    """Applies line slicing to file_read results when start_line/end_line are specified."""
    start_line = params.get("start_line")
    end_line = params.get("end_line")
    if start_line is None and end_line is None:
        return result_content

    try:
        lines = result_content.splitlines()
        if not lines:
            return ""

        s_line = max(1, int(start_line)) if start_line is not None else 1
        e_line = int(end_line) if end_line is not None else len(lines)
        s_line = min(s_line, len(lines))
        e_line = max(s_line, min(e_line, len(lines)))
        return "\n".join(lines[s_line - 1:e_line])
    except (ValueError, TypeError) as e:
        logger.error("Failed to parse line slice params: %s", e)
        return f"Error: Failed to slice file content: {e}"
