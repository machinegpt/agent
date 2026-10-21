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

import hashlib
import json
import logging
import os
import re
import sys
import textwrap
import signal
import time
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import yaml
import threading
import queue as _queue

from . import prompts
from .prompts import SYSTEM_PROMPT, TOOL_DEPTH_CRITICAL_MSG, construct_round_prompt
from .state import (
    AGENT_SET_KEY, CONFIG_MIN_KEY, is_count, merge_state, read_jinx, write_jinx,
)
from .tools import tool_schema
from . import evidence, learning, selfpatch

logger = logging.getLogger("jinx.runner")

HARD_CAP: int = 40
TOOL_DEPTH_CAP: int = 20

# Adaptive round count.
#
# How many rounds the loop requires before it will consider an exit is not a
# number the model sets once and walks away from. It is recomputed every round,
# from what the runner actually observed, and it only ever goes UP: a model that is
# tired, hurried or simply wrong has no way to make the loop shorter, which is the
# one direction in which laziness pays. What can move it:
#
#   * the model's own sizing (``protocol.loop.min``), accepted freely through round
#     ``SIZING_ROUNDS`` and ratcheted afterwards, so it can be raised but not lowered;
#   * ``DEAD_ROUND_CAP`` at most extra rounds, one per round that ran nothing new
#     while work was due (such a round does not count and cannot end the run);
#   * one confirming round (``CONFIRM_ROUNDS``) after the last failed round, so a
#     pass that arrives right after a fix has to be re-verified before it counts.
#
# ``--min`` pins the sizing to an exact value and wins over the model; the other
# two still apply on top of it, because "at least N" is what the flag promises.
# The extra rounds are bounded and never push the count past HARD_CAP - 1, so an
# exit stays reachable. These are brake logic: selfpatch protects them.
DEFAULT_MIN_ROUNDS: int = 10
SIZING_ROUNDS: int = 2
CONFIRM_ROUNDS: int = 1
DEAD_ROUND_CAP: int = 4
# Identical (call, result) pairs before the model is told it is repeating itself,
# and how many of them the ledger remembers (bounded: it lives in the run state).
REPEAT_NOTICE_AFTER: int = 3
CALL_LEDGER_CAP: int = 48

# IPC configuration (can be overridden via environment)
IPC_TIMEOUT = int(os.getenv("JINX_IPC_TIMEOUT", "10"))
IPC_RETRIES = int(os.getenv("JINX_IPC_RETRIES", "3"))
IPC_BACKOFF = float(os.getenv("JINX_IPC_BACKOFF", "1.0"))
BACKGROUND_WAIT_TIMEOUT = int(os.getenv("JINX_BACKGROUND_WAIT_TIMEOUT", "30"))


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
        """Serializes structures to YAML strings using the isolated dumper.

        Blank lines are preserved verbatim: they occur inside literal block
        scalars (``str_presenter`` renders multi-line strings with ``style='|'``)
        where removing them would change the value, so no blank-line rewriting
        is attempted. Compaction of the state block happens upstream in
        ``state.merge_scores`` / ``state.merge_text_list``, where it is lossless.
        """
        try:
            raw = yaml.dump(
                data, Dumper=Dumper, allow_unicode=True,
                default_flow_style=False, sort_keys=False, width=width
            )
            return raw if raw.endswith('\n') or not raw else raw + '\n'
        except Exception as e:
            raise SerializationError(f"Failed to serialize YAML string: {e}") from e

    @staticmethod
    def safe_atomic_write(path: Path, data: Any, width: int = sys.maxsize) -> None:
        """Writes data to files atomically via temporary staging files.

        Post-processes YAML to remove blank lines for compact output.
        """
        temp_path = None
        try:
            # Serialize using the canonical dump helper and clean blank lines.
            clean_yaml = Yaml.dump_to_string(data, width=width)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(clean_yaml)
            temp_path.replace(path)
        except Exception as e:
            logger.error("Atomic write failed on %s: %s", path, e, exc_info=True)
            try:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise IPCError(f"Atomic write failure on {path.name}: {e}") from e

    @staticmethod
    def load_from_file(path: Path) -> Any:
        """Safely loads and parses YAML structures from disk."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise SerializationError(f"Failed to load YAML file at {path}: {e}") from e


def parse_state_block(text: str) -> Optional[Dict[str, Any]]:
    """Extracts and parses the JINX state block from markdown code fences."""
    code_block_pattern = r"[ \t]*```(?:json|yaml|yml)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
    code_matches = list(re.finditer(code_block_pattern, text, re.DOTALL))

    state_keys = {"task", "facts", "scores", "debt", "open", "exit_ready", "deadlock"}
    # Strong markers — these rarely appear in prose so any one of them is a
    # reliable signal that the block is a real state update. We deliberately
    # exclude "task" and "open" from this set because they are common English
    # words and would cause false positives when LLMs embed explanatory YAML.
    strong_marker_keys = {"scores", "facts", "debt", "exit_ready", "deadlock"}

    def _looks_like_state(d: Dict[str, Any]) -> bool:
        keys = set(d.keys()) & state_keys
        # Either has the canonical "state:" wrapper, or contains at least one
        # strongly identifying marker. Bare "task" / "open" alone is not enough
        # because they appear in normal prose.
        return bool(d.get("state") or (keys & strong_marker_keys))

    if code_matches:
        for match in reversed(code_matches):
            raw = textwrap.dedent(match.group(1)).strip()
            try:
                data = yaml.safe_load(raw)
                if isinstance(data, dict):
                    nested = data.get("state")
                    if _looks_like_state(data):
                        return data
                    if isinstance(nested, dict) and _looks_like_state(nested):
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
            "content": prompts.MALFORMED_TOOL_BLOCK_MSG
        }

    if params is None:
        params = {}

    if not isinstance(params, dict):
        logger.error("Malformed tool_use block input: %r", params)
        return None, None, None, {
            "type": "tool_result", "tool_use_id": tool_use_id,
            "content": prompts.INVALID_TOOL_BLOCK_INPUT_MSG
        }

    return tool_use_id, name, params, None


def check_exit(scores: List[Dict[str, Any]], min_rounds: int, rnd: int) -> bool:
    """Evaluates whether the cognitive loop is ready to terminate."""
    if rnd < min_rounds:
        return False
    if not scores or len(scores) < 2:
        return False

    latest = scores[-1]
    if not _get_val(latest, "all_pass", False):
        return False

    if len(scores) >= 4:
        last3_best = max(_get_val(s, "pass_count", 0) for s in scores[-3:])
        prior_history = scores[:-3]
        if prior_history:
            prior_best = max((_get_val(s, "pass_count", 0) for s in prior_history), default=0)
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


def _extract_graph_data(g: Any) -> Optional[Dict[str, Any]]:
    """Extracts graph data from a dict or Pydantic model."""
    if g is None:
        return None
    if hasattr(g, "model_dump"):
        return g.model_dump()
    if isinstance(g, dict):
        return g
    return None


def _node_ids(g: Dict[str, Any]) -> set:
    """Returns the set of normalized node IDs from a graph dict."""
    raw = g.get("nodes")
    if not isinstance(raw, list):
        return set()
    return {
        str(n.get("id", "")).strip().lower()
        for n in raw if isinstance(n, dict) and n.get("id")
    }


def _edge_keys(g: Dict[str, Any]) -> set:
    """Returns the set of (source, relation, target) tuples from a graph dict."""
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


def _are_approaches_similar(entry1: Any, entry2: Any) -> bool:
    """Calculates semantic similarity between two approach graphs or falls back to text-matching."""
    graph1 = _get_val(entry1, "approach_graph")
    graph2 = _get_val(entry2, "approach_graph")

    g1 = _extract_graph_data(graph1)
    g2 = _extract_graph_data(graph2)

    if not g1 or not g2:
        return _get_val(entry1, "approach", "") == _get_val(entry2, "approach", "")

    nodes1, nodes2 = _node_ids(g1), _node_ids(g2)
    edges1, edges2 = _edge_keys(g1), _edge_keys(g2)
    if not nodes1 and not edges1 and not nodes2 and not edges2:
        return _get_val(entry1, "approach", "") == _get_val(entry2, "approach", "")
    # Guard against division by zero when one side is entirely empty
    node_sim = len(nodes1.intersection(nodes2)) / len(nodes1.union(nodes2)) if nodes1.union(nodes2) else None
    edge_sim = len(edges1.intersection(edges2)) / len(edges1.union(edges2)) if edges1.union(edges2) else None
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
        raise IPCError(f"Failed to transmit tool payload: {e}") from e

    # Await stdin with timeout to avoid indefinite blocking when editor doesn't reply.
    try:
        line = _read_stdin_with_retries(IPC_TIMEOUT, IPC_RETRIES, IPC_BACKOFF)
        if line is None:
            raise IPCError(f"Timeout waiting for editor response after {IPC_TIMEOUT}s x{IPC_RETRIES} attempts")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as e:
            raise IPCError(f"Malformed JSON from editor: {e}") from e

        status = response.get("status", "")
        is_error = "error" in response or (isinstance(status, str) and "error" in status.lower())
        output = response.get("output") if "output" in response else (response.get("content") if "content" in response else (response.get("error") if "error" in response else str(response)))
        was_sliced = bool(response.get("sliced") or response.get("is_sliced"))
        return str(output), was_sliced, is_error
    except OSError as e:
        raise IPCError(f"Error receiving input: {e}") from e


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
        line = _read_stdin_with_retries(IPC_TIMEOUT, IPC_RETRIES, IPC_BACKOFF)
        if line is None:
            raise IPCError(f"Timeout waiting for LLM response from editor after {IPC_TIMEOUT}s x{IPC_RETRIES} attempts")
        data = json.loads(line)
        content = data.get("content") or []
        if not isinstance(content, list):
            raise IPCError("Invalid LLM response content format (expected list)")
        return content
    except (json.JSONDecodeError, OSError, IPCError) as e:
        logger.error("Error receiving LLM response: %s", e)
        raise IPCError(f"Error receiving LLM response: {e}") from e


AGENT_DIR: Path = Path(__file__).resolve().parent.parent.parent
REQUEST_PATH: Path = AGENT_DIR / "jinx_request.yaml"
RESPONSE_PATH: Path = AGENT_DIR / "jinx_response.yaml"
RUN_STATE_PATH: Path = AGENT_DIR / "jinx_run_state.yaml"


def clean_up_ipc_files() -> None:
    """Removes temporary IPC communication files."""
    for p in (REQUEST_PATH, RESPONSE_PATH, RUN_STATE_PATH):
        p.unlink(missing_ok=True)


# AGENT_DIR is derived from this module's location, not from the CWD, so the IPC
# paths above are shared by *every* process that imports jinx. An editor, a
# linter or a test runner that merely imports the package and then receives
# Ctrl+C would delete a live session's request, response and run state - the
# session it has no part in, from whatever directory it happens to be in. Only
# the process that actually wrote those files may remove them.
_IPC_OWNER = False


def _claim_ipc_files() -> None:
    """Record that this process created the IPC files and may clean them up."""
    global _IPC_OWNER
    _IPC_OWNER = True


# Register cleanup handlers to ensure IPC files are removed on exit/signals.
def _signal_cleanup(signum=None, frame=None) -> None:
    try:
        logger.info("Signal %s received: cleaning up IPC files.", signum)
    except Exception:
        pass
    # Escape hatch: keep the IPC files so a run interrupted mid-round can be
    # inspected or resumed. Deleting them on Ctrl+C destroys the only record of
    # what the model was last told, which is usually exactly what you need when
    # a loop is misbehaving. Set JINX_KEEP_IPC_ON_SIGNAL=1 to preserve them.
    keep = os.environ.get("JINX_KEEP_IPC_ON_SIGNAL", "").strip().lower() in ("1", "true", "yes", "on")
    if keep:
        try:
            logger.warning(
                "JINX_KEEP_IPC_ON_SIGNAL set: preserving IPC files (request/run-state) "
                "for inspection."
            )
        except Exception:
            pass
    else:
        if _IPC_OWNER:
            try:
                clean_up_ipc_files()
            except Exception:
                pass
        else:
            # Not a failure: a bystander process (editor, linter, test runner)
            # was interrupted, and the session files belong to a run that is
            # still going. Deleting them here would destroy that run's only
            # record of the current round.
            try:
                logger.warning(
                    "Signal %s received in a process that does not own the JINX "
                    "IPC files; leaving %s intact.", signum, AGENT_DIR,
                )
            except Exception:
                pass
    # Use os._exit to avoid sys.exit raising SystemExit inside a signal handler,
    # which can cause recursion if the handler itself was invoked during cleanup.
    try:
        os._exit(1)
    except Exception:
        pass


def _install_signal_handlers() -> None:
    """Installs the IPC cleanup handlers for the signals a run can receive.

    Registration is deliberately deferred to the run entry points instead of
    happening at import time. This module is imported by test runners, editor
    integrations and lint tooling, and the handler ends the process with
    ``os._exit`` while deleting the IPC files. Installed on import, it silently
    disarmed Ctrl+C in every one of those processes and removed files they had
    no business removing; the agent is supposed to be stopped, not to vanish.
    """
    # SIGBREAK is the one Windows actually adds (Ctrl+Break); SIGHUP is the one
    # POSIX hosts have and Windows lacks. Listing both keeps each platform's
    # interrupt path registered instead of relying on getattr to fail.
    for sig in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
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


def _loop_of(jinx: Dict[str, Any]) -> Dict[str, Any]:
    """The ``protocol.loop`` mapping stored in the manifest, or a throwaway ``{}``."""
    protocol = jinx.get("protocol")
    if isinstance(protocol, dict):
        loop = protocol.get("loop")
        if isinstance(loop, dict):
            return loop
    return {}


def _scores_of(jinx: Dict[str, Any]) -> List[Any]:
    """A copy of the stored score list, or ``[]`` when the manifest has none."""
    state = jinx.get("state")
    scores = state.get("scores") if isinstance(state, dict) else None
    return list(scores) if isinstance(scores, list) else []


def _discard_sizing(jinx: Dict[str, Any]) -> None:
    """Drops the round count the previous task's model wrote, keeping a configured one.

    ``merge_state`` writes the model's ``protocol.loop.min`` into the manifest, and
    the manifest outlives the task: left in place, a count sized for a one-line
    question would silently become the floor of the next, harder task (or the
    reverse). Only a value the model wrote is dropped; a value a person configured
    is restored from ``config_min`` if sizing had replaced it, and is otherwise
    never touched.
    """
    loop = _loop_of(jinx)
    if not loop.get(AGENT_SET_KEY):
        return
    configured = loop.pop(CONFIG_MIN_KEY, None)
    if is_count(configured):
        loop["min"] = configured
    else:
        loop.pop("min", None)
    loop.pop(AGENT_SET_KEY, None)


def _init_new_session(task: str, jinx: Dict[str, Any]) -> None:
    """Initializes a fresh task session in the JINX state."""
    if not isinstance(jinx.get("state"), dict):
        jinx["state"] = {}
    jinx["state"].update({
        "task": task, "facts": [], "scores": [], "debt": [],
        "open": [], "exit_ready": False, "deadlock": False
    })
    _discard_sizing(jinx)
    write_jinx(jinx)
    # NOTE: the durable lesson ledger in .agent/lessons.yaml is deliberately left
    # untouched. A new task wipes working memory, but what was learned is exactly
    # what should carry over; resetting it here would make the store decay at the
    # same rate as the state it is supposed to outlast.


# ------------------------------------------------------------------------------
# Adaptive round count: the ledger and the rules that read it.
#
# The ledger lives in the run state (``run_state["rounds"]``) because every round
# is a new process, and it is a handful of integers plus one bounded map, so the
# run state stays flat however long the task runs. Nothing in it is read from the
# model: it is built from what the runner itself saw (the tool calls, what they
# returned, which rounds ended in which verdict).
# ------------------------------------------------------------------------------

# Run-time noise that makes two identical outcomes look different: durations,
# addresses and timestamps. Stripped before a result is fingerprinted, so a
# test run that fails the same way twice is recognised as doing so. It is
# deliberately narrow: digits in general are NOT removed, because "3 failed" and
# "2 failed" are exactly the difference this has to keep.
_RESULT_NOISE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds)\b|0x[0-9a-fA-F]+"
    r"|\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?Z?"
)


def _new_ledger() -> Dict[str, Any]:
    """An empty round ledger."""
    return {
        "declared": None, "shown": None, "dead": 0, "dead_streak": 0,
        "last_fail": 0, "round_calls": 0, "round_novel": 0, "calls": {},
    }


def _ledger_of(run_state: Dict[str, Any]) -> Dict[str, Any]:
    """The run state's round ledger, validated, attached to the run state and returned.

    A ledger read back from disk is as untrusted as the run state around it: a
    truncated, hand-edited or older file has to yield an empty ledger, not a crash
    in the middle of a round, and the call memory is clipped to its bound.
    """
    ledger = _new_ledger()
    raw = run_state.get("rounds")
    if isinstance(raw, dict):
        for key in ("dead", "dead_streak", "last_fail", "round_calls", "round_novel"):
            value = raw.get(key)
            ledger[key] = value if is_count(value) and value >= 0 else 0
        for key in ("declared", "shown"):
            value = raw.get(key)
            ledger[key] = value if is_count(value) and value >= 0 else None
        calls = raw.get("calls")
        if isinstance(calls, dict):
            for key, entry in list(calls.items())[-CALL_LEDGER_CAP:]:
                if (
                    isinstance(key, str) and isinstance(entry, list) and len(entry) == 3
                    and isinstance(entry[0], str) and is_count(entry[1]) and is_count(entry[2])
                ):
                    ledger["calls"][key] = [entry[0], entry[1], entry[2]]
    run_state["rounds"] = ledger
    return ledger


def _digest(text: str) -> str:
    """A short, stable fingerprint. Not a security boundary, only an identity."""
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=6).hexdigest()


def _call_fingerprint(name: Any, params: Any) -> str:
    """Identifies a tool call by its name and its full, canonicalised input."""
    try:
        body = json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        body = repr(params)
    return _digest("%s\n%s" % (name, body))


def _result_text(content: Any) -> str:
    """A tool result as text with run-time noise removed, ready to be fingerprinted."""
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        try:
            text = Yaml.dump_to_string(content)
        except JinxError:
            text = repr(content)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _RESULT_NOISE.sub("#", text)


def _call_label(call: Dict[str, Any]) -> str:
    """A short, readable name for a call, for the notice that reports a repeat."""
    params = call.get("params")
    detail = ""
    if isinstance(params, dict):
        detail = str(params.get("script") or params.get("path") or "")
    detail = " ".join(detail.split())
    if len(detail) > 60:
        detail = detail[:57] + "..."
    name = str(call.get("name") or "tool")
    return "%s `%s`" % (name, detail) if detail else name


def _tool_calls_in(history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Every well-formed ``tool_use`` block in the history window, keyed by its id.

    The results of a tool batch arrive in a later process, so the call that
    produced a result is found again here, in the history the run state carries.
    """
    found: Dict[str, Dict[str, Any]] = {}
    for msg in history or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_use_id, name, params, err = _validate_tool_use_block(block)
                if err is None and tool_use_id is not None:
                    found[tool_use_id] = {"id": tool_use_id, "name": name, "params": params}
    return found


def _record_tool_results(
    ledger: Dict[str, Any], calls_by_id: Dict[str, Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """Notes what a tool batch returned and reports the calls that are only repeating.

    A result is *new* when its call was never seen, or was seen but returned
    something different. It is a *repeat* when the same call returned the same
    outcome again, and it is reported once it has done so ``REPEAT_NOTICE_AFTER``
    times. A result that cannot be matched to the call that produced it counts as
    new: a round is only ever called dead on positive evidence, never for lack of it.

    Returns ``(label, times)`` for each call that has reached the threshold.
    """
    repeats: List[Tuple[str, int]] = []
    seen = ledger["calls"]
    for result in results or []:
        if not isinstance(result, dict):
            continue
        ledger["round_calls"] += 1
        tool_use_id = result.get("tool_use_id")
        call = calls_by_id.get(tool_use_id) if isinstance(tool_use_id, str) else None
        if call is None:
            ledger["round_novel"] += 1
            continue
        key = _call_fingerprint(call.get("name"), call.get("params"))
        outcome = _digest(_result_text(result.get("content")))
        previous = seen.pop(key, None)
        total, same = 1, 1
        if previous is not None:
            total = previous[2] + 1
            if previous[0] == outcome:
                same = previous[1] + 1
        seen[key] = [outcome, same, total]
        if same == 1:
            ledger["round_novel"] += 1
        elif same >= REPEAT_NOTICE_AFTER:
            repeats.append((_call_label(call), same))
    while len(seen) > CALL_LEDGER_CAP:
        seen.pop(next(iter(seen)))
    return repeats


def _finish_round(ledger: Dict[str, Any], prior_scores: List[Any]) -> bool:
    """Closes the tool tally of the round that just ended; True when it was dead.

    A round is dead when work was due and none was done. Work is due in two
    situations: the last recorded round failed (a *fixing* round, dead if nothing
    NEW was run, edited or read: the same commands returning the same results are
    a model going through the motions), or the last round passed straight after a
    failure (a *confirming* round, dead only if it ran no command at all, so that
    re-running the project's own tests still counts as confirming). Round 1 and
    rounds after a clean pass are never dead: nothing was owed.

    The tally is reset here, so the next round starts counting from zero.
    """
    fixing = bool(prior_scores) and not _get_val(prior_scores[-1], "all_pass", False)
    confirming = (
        not fixing and len(prior_scores) >= 2
        and not _get_val(prior_scores[-2], "all_pass", False)
    )
    if fixing:
        dead = ledger["round_novel"] == 0
    elif confirming:
        dead = ledger["round_calls"] == 0
    else:
        dead = False
    if dead:
        ledger["dead"] += 1
        ledger["dead_streak"] += 1
    else:
        ledger["dead_streak"] = 0
    ledger["round_calls"] = 0
    ledger["round_novel"] = 0
    return dead


def _note_round_result(ledger: Dict[str, Any], scores: List[Any], rnd: int) -> None:
    """Records the round a failure happened in, by the runner's own round counter.

    The model's ``round`` numbers are not used: they are written by the model, and
    the confirming round has to be measured from the failure the runner saw.
    """
    if scores and not _get_val(scores[-1], "all_pass", False):
        ledger["last_fail"] = rnd


def _declared_base(
    jinx: Dict[str, Any], min_override: Optional[int], ledger: Dict[str, Any], rnd: int
) -> int:
    """The round count in force before any runner-added rounds. It only ever goes up.

    ``--min`` pins it exactly. Otherwise the model's ``protocol.loop.min`` (marked
    ``agent_set`` by ``merge_state``) sizes the loop: the first value it sends
    through round ``SIZING_ROUNDS`` is taken as it is, because sizing a small task
    below the default is the point of sizing; a first value sent later can only
    raise the count, and every later value is ratcheted with ``max``. Without a
    value from the model the count is the configured one, else the default.

    The value is clamped to ``[2, HARD_CAP - 1]``: nothing can exit before round 2,
    and a count the loop can never reach would just burn rounds to the hard cap.
    """
    if min_override is not None:
        return min_override
    loop = _loop_of(jinx)
    if loop.get(AGENT_SET_KEY):
        configured, sent = loop.get(CONFIG_MIN_KEY), loop.get("min")
    else:
        configured, sent = loop.get("min"), None
    if not is_count(configured):
        configured = DEFAULT_MIN_ROUNDS
    declared = ledger.get("declared")
    if is_count(sent):
        sent = max(2, min(sent, HARD_CAP - 1))
        if declared is None:
            declared = sent if rnd <= SIZING_ROUNDS else max(configured, sent)
        else:
            declared = max(declared, sent)
    ledger["declared"] = declared
    return declared if declared is not None else configured


def _store_declared(jinx: Dict[str, Any], min_override: Optional[int], base: int) -> None:
    """Writes the ratcheted count back, so the manifest shows what is in force.

    Without it a later, lower value from the model would sit in the manifest while
    the runner ignored it, and anything that reads ``JINX.yaml`` would see a count
    the loop is not using. A pinned run leaves the manifest alone.
    """
    if min_override is not None:
        return
    loop = _loop_of(jinx)
    if loop.get(AGENT_SET_KEY) and loop.get("min") != base:
        loop["min"] = base


def _round_floors(base: int, ledger: Dict[str, Any]) -> Tuple[int, int]:
    """``(floor, exit_floor)``: the round before which a deadlock, resp. an exit, is not considered.

    ``floor`` is the sized count plus one round per dead round, at most
    ``DEAD_ROUND_CAP`` of them. ``exit_floor`` also requires ``CONFIRM_ROUNDS`` more
    rounds after the last failed one. It applies to an exit only: a deadlock is, by
    definition, a run that is still failing, so a floor that moved with every failure
    would put it forever out of reach.

    The extra rounds are capped at ``HARD_CAP - 1`` so an exit is always reachable,
    and the result is never below ``base``, so a pinned ``--min`` is never undercut.
    """
    ceiling = max(base, HARD_CAP - 1)
    floor = min(base + min(ledger["dead"], DEAD_ROUND_CAP), ceiling)
    exit_floor = floor
    if ledger["last_fail"]:
        exit_floor = max(floor, min(ledger["last_fail"] + 1 + CONFIRM_ROUNDS, ceiling))
    return floor, exit_floor


def _close_round(
    jinx: Dict[str, Any], update: Optional[Dict[str, Any]], ledger: Dict[str, Any],
    min_override: Optional[int], rnd: int, diagnostics: List[str],
) -> Dict[str, Any]:
    """Merges the reply's state block and re-derives the round count for the round that ended.

    Shared by both transports so the rules cannot drift apart. The order matters:
    the round is classified against the scores as they stood BEFORE this reply's
    entry, because that is what decided whether work was due; only then is the
    block merged, the sized count ratcheted and the floors recomputed.
    """
    dead = _finish_round(ledger, _scores_of(jinx))
    outcome: Dict[str, Any] = {}
    if update:
        jinx = merge_state(jinx, update, diagnostics=diagnostics, outcome=outcome)
    base = _declared_base(jinx, min_override, ledger, rnd)
    if update:
        _store_declared(jinx, min_override, base)
        write_jinx(jinx)
    scores = _scores_of(jinx)
    applied = bool(outcome.get("applied"))
    if applied:
        _note_round_result(ledger, scores, rnd)
    floor, exit_floor = _round_floors(base, ledger)
    return {
        "jinx": jinx, "outcome": outcome, "applied": applied, "scores": scores,
        "dead": dead, "base": base, "floor": floor, "exit_floor": exit_floor,
    }


def _round_verdict(
    jinx: Dict[str, Any], scores: List[Any], ledger: Dict[str, Any], dead: bool,
    floor: int, exit_floor: int, rnd: int, diagnostics: List[str],
) -> str:
    """``"exit"``, ``"deadlock"`` or ``""`` for a round whose block was accepted.

    Exit needs ``exit_ready``, ``check_exit`` at the confirmed count and a round
    that was not dead (a dead round is bounded by ``DEAD_ROUND_CAP``, after which
    it is tolerated, so a task with nothing new left to run can still finish).
    A deadlock needs the evidence ``check_deadlock`` looks for, or a claim made from
    ``floor`` on. A claim made earlier is refused, its flag cleared and the model
    told: stopping early is the one request the loop never grants on request alone.
    """
    flags = jinx.setdefault("state", {})
    counted_dead = dead and ledger["dead"] <= DEAD_ROUND_CAP
    if flags.get("exit_ready") and not counted_dead and check_exit(scores, exit_floor, rnd):
        return "exit"
    claim = bool(flags.get("deadlock"))
    if check_deadlock(scores, floor, rnd) or (claim and rnd >= floor):
        if not claim:
            flags["deadlock"] = True
            write_jinx(jinx)
        return "deadlock"
    if claim:
        flags["deadlock"] = False
        write_jinx(jinx)
        diagnostics.append(prompts.DEADLOCK_REFUSED % {"rnd": rnd, "floor": floor})
    return ""


def _adaptive_info(
    jinx: Dict[str, Any], base: int, ledger: Dict[str, Any], floor: int,
    exit_floor: int, dead: bool, next_round: int, min_override: Optional[int],
) -> Dict[str, Any]:
    """The runner-observed facts the next prompt reports (see ``prompts.note_adaptive``)."""
    configured = _loop_of(jinx).get("min") if not _loop_of(jinx).get(AGENT_SET_KEY) else None
    return {
        "shown": ledger.get("shown"),
        "base": base,
        "dead": min(ledger["dead"], DEAD_ROUND_CAP),
        "dead_streak": ledger["dead_streak"],
        "last_dead": dead,
        # The round that just ended failed: the count moving with it is not news, the
        # fix is. Without this a run that keeps failing is told "raised" every round.
        "failing": ledger["last_fail"] == next_round - 1,
        "failed_round": ledger["last_fail"] if exit_floor > floor else 0,
        "unsized": (
            min_override is None and ledger.get("declared") is None
            and not is_count(configured) and next_round <= SIZING_ROUNDS
        ),
    }


def _inject_lessons(run_state: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Renders the bounded LEARNED RULES block and records what was injected.

    Returns the block text and the normalized keys of the lessons actually shown,
    which the caller must pass to :func:`write_llm_request` so they survive the
    next process. The NEXT round then credits or blames exactly those lessons,
    and no others. Without that round-trip the ledger could not tell a proven
    rule from a useless one.
    """
    if not selfpatch.SELF_PATCH_ENABLED:
        return "", []
    try:
        ledger = learning.load_ledger()
        rendered = learning.render_lessons(ledger.get("lessons"))
    except Exception as e:  # never let the learning store break a round
        logger.error("Lesson injection failed: %s", e, exc_info=True)
        return "", []
    run_state["applied_lessons"] = rendered["applied"]
    return rendered["text"], rendered["applied"]


def _close_lesson_bookkeeping(
    run_state: Dict[str, Any], passed: Optional[bool]
) -> None:
    """Credits or blames the lessons that were injected into the finished round."""
    applied = run_state.get("applied_lessons") or []
    if not applied:
        return
    try:
        ledger = learning.load_ledger()
        ledger["lessons"] = learning.record_outcome(
            ledger.get("lessons"), applied, bool(passed)
        )
        learning.save_ledger(ledger)
    except Exception as e:
        logger.error("Lesson bookkeeping failed: %s", e, exc_info=True)
    finally:
        run_state.pop("applied_lessons", None)


def _rollback_and_report(reason: str) -> str:
    """Rolls the self-patch back and builds the model's refusal message.

    The rollback is reported separately from the reason because the two can
    disagree: a partially failed restore leaves files on disk that the model
    believes were undone, and saying so is the only way it can find out.
    """
    try:
        restored = selfpatch.restore_baseline()
    except Exception as e:
        logger.error("Self-patch rollback failed: %s", e, exc_info=True)
        return prompts.SELF_PATCH_ROLLBACK_FAILED % (reason, type(e).__name__, e)
    logger.warning("Self-patch protection gate reverted %d file(s)", len(restored))
    return prompts.SELF_PATCH_REFUSED % (reason, ", ".join(restored) or "none")


def _enforce_self_patch_gate(run_state: Dict[str, Any]) -> Optional[str]:
    """Verifies or reverts self-edits. Returns feedback for the model, or None.

    Returning the message rather than pushing it into ``history`` and a carried
    ``self_patch_feedback`` field keeps a single delivery path: the caller
    appends it once, after the tool results, so the model reads it in the order
    the protocol requires and cannot receive it twice.
    """
    if not selfpatch.SELF_PATCH_ENABLED:
        return None
    if not selfpatch.BASELINE_DIR.exists():
        return None
    try:
        changed = selfpatch.baseline_changed()
    except Exception as e:
        logger.error("Self-patch diff failed: %s", e, exc_info=True)
        return None
    if not changed:
        # Keep the baseline: it is the reference for the whole run, not a
        # one-shot checkpoint. Clearing it on an unchanged round would disarm
        # the gate before any tool call had a chance to break something.
        return None

    # Re-check protection against the baseline before verifying. `guard_tool_call`
    # only ever sees `file_write`, so a brake weakened through `bash_exec` would
    # otherwise sail through a green test suite and be adopted as the baseline.
    try:
        for name in changed:
            # 1. Block any edit to PROTECTED_FILES
            if name in selfpatch.PROTECTED_FILES:
                return _rollback_and_report(
                    prompts.PROTECTED_FILE_REFUSAL % name
                )

            # 2. Check for symbol violations in other files.
            # We must treat the change as a whole-file rewrite to detect
            # deletions or shadowing of protected symbols.
            base_path = selfpatch._target_for(name)
            old_text = base_path.read_text(encoding="utf-8", errors="replace")
            current_text = selfpatch.snapshot().get(name, "")

            violations = selfpatch._violations_against(
                name, current_text, old_text, whole_file=True
            )
            if violations:
                detail = "; ".join(violations)
                return _rollback_and_report(
                    prompts.PROTECTED_SYMBOL_REFUSAL % (name, detail)
                )
    except Exception as e:
        # Fail closed. If the check itself cannot run, this round has no evidence
        # that the brake survived, and "no evidence of a violation" is not the
        # same as "no violation": continuing would let `capture_baseline` adopt
        # whatever is on disk as the new trusted reference.
        logger.error("Self-patch protection check failed: %s", e, exc_info=True)
        return _rollback_and_report(
            prompts.PROTECTION_CHECK_FAILURE % (type(e).__name__, e)
        )

    # Put the tests and test configuration back BEFORE the suite runs. They are
    # the yardstick the patch is about to be measured with, so leaving a weakened
    # or skipped test in place would let the model edit the answer instead of the
    # framework. Source changes are deliberately left alone here: they are the
    # thing under test, and a failing patch is reverted below anyway.
    try:
        tests_restored = selfpatch.restore_baseline(only=selfpatch.REPO_PREFIX)
    except Exception as e:
        logger.error("Could not restore the test files: %s", e, exc_info=True)
        return _rollback_and_report(
            prompts.TEST_SUITE_RESTORE_FAILURE % (type(e).__name__, e)
        )

    result = selfpatch.verify(AGENT_DIR.parent)
    if result["ok"]:
        logger.info("Self-patch verified: %s", ", ".join(changed))
        # Adopt the verified edit as the new reference so the same diff is not
        # re-verified on every later round.
        selfpatch.capture_baseline()
        if tests_restored:
            # The tests were put back before verification, so a new or edited test
            # file the model added is now gone. Staying silent would leave it
            # believing the coverage it just wrote still exists.
            return prompts.TEST_FILES_RESTORED % ", ".join(tests_restored)
        return None

    # Route the rollback through the shared handler. `restore_baseline` reads
    # every baseline file, and a source file that is no longer valid UTF-8 raises
    # UnicodeDecodeError -- which is a ValueError, not an OSError, so it escapes
    # the (IPCError, OSError, JinxError) guard in `run_file_ipc` and kills the run
    # with no message to the model at all. The refusal branches already prove the
    # shared path is the correct one here.
    revert_notice = prompts.SELF_PATCH_REVERTED % (
        ", ".join(changed),
        result["summary"],
        _render_check_failures(result["checks"]),
    )
    return _rollback_and_report(revert_notice)


DEFAULT_RAW_TAIL_LINES: int = 12


def _elide_tail(tail: str, max_lines: int = DEFAULT_RAW_TAIL_LINES) -> str:
    """Shortens a captured tail to its head and tail, marking what was dropped.

    The tail arrives already truncated to 25 lines from the end, so the head is
    where the traceback detail lives. Keeping the first and last few lines and
    eliding the middle bounds the notice while preserving both the traceback and
    the summary line that identifies the failures.
    """
    lines = [evidence.strip_noise(l) for l in tail.rstrip().splitlines()]
    lines = [l for l in lines if l.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = max(1, max_lines // 2)
    tail_lines = max(1, max_lines - head)
    dropped = len(lines) - head - tail_lines
    middle = [prompts.HISTORY_ELISION_MARKER % dropped]
    return "\n".join(lines[:head] + middle + lines[-tail_lines:])


def _render_check_failures(checks: List[Dict[str, Any]]) -> str:
    """Renders failed checks grouped by cause instead of as raw tails.

    A tail is a symptom list capped at 25 lines and truncated from the start, so
    one broken helper arrives as dozens of near-identical lines with the common
    cause out of view. Grouping collapses that to the file, the cause and how
    many tests share it.

    A bounded, elided tail is still kept underneath: grouping infers a cause from
    message text, and keeping the original lets the model check that inference
    instead of trusting it. It is elided rather than pasted whole because the
    grouping already covers what the repetition was used for.
    """
    lines: List[str] = []
    for check in checks:
        if check.get("ok"):
            continue
        name = str(check.get("name") or "?")
        tail = str(check.get("tail") or "")
        digest = evidence.evidence_from_check(check)
        if digest:
            lines.append(digest)
        else:
            lines.append(
                prompts.CHECK_FAILURE_LINE % (name, tail or "(no output captured)")
            )
        if tail.strip():
            lines.append("[%s] raw tail (untruncated evidence):" % name)
            if digest:
                lines.append(_elide_tail(tail))
            else:
                lines.append(tail)
    if not lines:
        lines.append(prompts.CHECK_FAILURE_LINE % ("?", "verification failed with no output"))
    return "\n".join(lines)


def _finish_run() -> None:
    """Terminal cleanup for a run that ended on its own terms.

    The self-patch baseline is dropped here because the run is over and the next
    task recaptures it. It is intentionally NOT dropped on the error exits: a run
    that died with a half-applied self-edit should leave the baseline behind so
    the bootstrap preflight can still roll that edit back on the next invocation.
    """
    if selfpatch.SELF_PATCH_ENABLED:
        selfpatch.clear_baseline()
    clean_up_ipc_files()


def _touch_run_state(run_state: Dict[str, Any]) -> Dict[str, Any]:
    """Marks a run state as recently updated so stale waits can recover."""
    run_state["updated_at"] = time.time()
    return run_state


def _is_run_state_stale(run_state: Dict[str, Any], timeout_seconds: int = BACKGROUND_WAIT_TIMEOUT) -> bool:
    """Returns True only when the session is truly idle and no active file activity is detected."""
    if not isinstance(run_state, dict):
        return False

    updated_at = run_state.get("updated_at")
    if isinstance(updated_at, (int, float)):
        if (time.time() - float(updated_at)) <= timeout_seconds:
            return False

    for candidate in (RUN_STATE_PATH, REQUEST_PATH):
        try:
            if candidate.exists() and (time.time() - candidate.stat().st_mtime) <= timeout_seconds:
                return False
        except OSError:
            continue

    return True


def _extract_last_tool_calls(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extracts the latest tool_use blocks that have no result yet.

    A call that already has a ``tool_result`` further down the history was answered
    without being dispatched (refused as a protected self-patch, or malformed), so
    re-issuing it would run exactly what the guard turned down.
    """
    answered = set()
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        calls: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                answered.add(block.get("tool_use_id"))
            elif block.get("type") == "tool_use":
                tool_use_id, name, params, err = _validate_tool_use_block(block)
                if (
                    err is None and tool_use_id is not None and name is not None
                    and tool_use_id not in answered
                ):
                    calls.append({"id": tool_use_id, "name": name, "params": params})
        if calls:
            return calls
    return []


HISTORY_WINDOW: int = int(os.getenv("JINX_HISTORY_WINDOW", "6"))
HISTORY_PERSIST_WINDOW: int = int(os.getenv("JINX_HISTORY_PERSIST_WINDOW", "8"))
# A past tool call can carry a whole file (``file_write``), and the window ships it
# again in every request until it scrolls out. Strings in the ``tool_use`` inputs of
# the REQUEST window are cut to this many characters; 0 disables the cut. The run
# state keeps full inputs, because a call that has to be re-issued needs them.
HISTORY_TOOL_INPUT_CHARS: int = int(os.getenv("JINX_HISTORY_TOOL_INPUT_CHARS", "1500"))
# Bound on the memoized tool results kept in the run state. Large enough to cover
# a deep tool loop, small enough that the cache never dominates the file.
TOOL_RESULT_CACHE_CAP: int = int(os.getenv("JINX_TOOL_RESULT_CACHE_CAP", "64"))


def _has_orphan_tool_result(msg: Dict[str, Any]) -> bool:
    """True when the message is a tool_result with no preceding tool_use.

    A history window can slice between a ``tool_use`` and its ``tool_result``.
    Sending an orphan ``tool_result`` to a chat-completions style API is a hard
    error, so the window is nudged forward until it starts on a safe message.
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )


def _has_unanswered_tool_use(msg: Dict[str, Any]) -> bool:
    """True when the message requests tools, i.e. its results come later."""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_use"
        for b in content
    )


def compact_history_for_request(
    history: List[Dict[str, Any]], max_messages: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Keeps only the most recent exchange, dropping orphaned tool blocks.

    The window is advanced past a leading ``tool_result`` block so the model
    never receives a result whose request is not in the window, and a trailing
    ``tool_use`` is cut because its result does not exist yet. This is the window
    that is sent; ``_persist_window`` bounds what is kept in the run state, which
    is what keeps that file from growing without limit across a long task.
    """
    limit = HISTORY_WINDOW if max_messages is None else max_messages
    if limit <= 0 or len(history) <= limit:
        window = list(history)
    else:
        window = list(history[-limit:])
    while window and _has_unanswered_tool_use(window[-1]):
        window = window[:-1]
    idx = 0
    # Advance past every orphaned tool_result, including a window that consists of
    # exactly one message: keeping the last element unconditionally would let a
    # lone tool_result reach the API with no tool_use to answer.
    while idx < len(window) and _has_orphan_tool_result(window[idx]):
        idx += 1
    return window[idx:]


def _persist_window(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The newest ``HISTORY_PERSIST_WINDOW`` messages, as kept in the run state.

    Unlike ``compact_history_for_request`` this keeps a trailing assistant
    ``tool_use``. While the loop waits for the host to run the tools, that message
    is the only record of what was asked: a stale-wait retry re-issues it
    (``_extract_last_tool_calls``) and the ``tool_result`` appended later has to
    follow it. Trimming it here, as the request window rightly does, left every
    later result without its request; each of them then counted as an orphan, and
    after a handful of tool replies the whole window was discarded. Only a leading
    orphan ``tool_result`` is dropped.
    """
    limit = HISTORY_PERSIST_WINDOW
    window = list(history[-limit:]) if 0 < limit < len(history) else list(history)
    idx = 0
    while idx < len(window) and _has_orphan_tool_result(window[idx]):
        idx += 1
    return window[idx:]


def _shrink_tool_inputs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copies ``messages`` with oversized ``tool_use`` string inputs shortened.

    The model wrote those inputs and the host already ran them, so a request only
    needs enough to recognise the call. The marker says how much was left out; the
    ``id``, ``name`` and ``input`` object stay valid. Nothing is mutated, so the
    history that is persisted and re-issued keeps the full inputs.
    """
    cap = HISTORY_TOOL_INPUT_CHARS
    if cap <= 0:
        return list(messages)
    shrunk: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            shrunk.append(msg)
            continue
        blocks: List[Any] = []
        changed = False
        for block in content:
            params = block.get("input") if isinstance(block, dict) else None
            if (
                isinstance(block, dict) and block.get("type") == "tool_use"
                and isinstance(params, dict)
                and any(isinstance(v, str) and len(v) > cap for v in params.values())
            ):
                block = dict(block, input={
                    k: v[:cap] + prompts.HISTORY_INPUT_TRUNCATED % (len(v) - cap)
                    if isinstance(v, str) and len(v) > cap else v
                    for k, v in params.items()
                })
                changed = True
            blocks.append(block)
        shrunk.append(dict(msg, content=blocks) if changed else msg)
    return shrunk


def summarize_dropped_history(
    dropped: List[Dict[str, Any]], kept: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Builds a one-message digest of history that fell outside the window.

    ``kept`` is the window the model actually receives, so ``dropped`` should be
    measured against that same window — otherwise the count describes what was
    written to disk rather than what the model was shown.

    Nothing is lost: the durable record lives in ``JINX.yaml`` via the score
    history, and this note tells the model how much earlier context was elided so
    it does not assume the window is the whole session.
    """
    if not dropped:
        return None
    rounds = [d for d in dropped if isinstance(d, dict)]
    tool_msgs = sum(1 for d in rounds if _has_unanswered_tool_use(d) or _has_orphan_tool_result(d))
    return {
        "role": "user",
        "content": prompts.HISTORY_ELISION_NOTICE % (len(rounds), tool_msgs),
    }


def _request_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ``messages`` of an ``llm_generate`` request: the send window plus a notice.

    Every such request is built here, the final no-tools one included, so the model
    is always told how much earlier context it is not seeing. Inputs of past tool
    calls are shortened (``_shrink_tool_inputs``). The notice is measured against
    the send window and is never persisted.
    """
    send_window = compact_history_for_request(history)
    kept = {id(m) for m in send_window}
    dropped = [m for m in history if id(m) not in kept]
    messages = _shrink_tool_inputs(send_window)
    notice = summarize_dropped_history(dropped, send_window)
    if notice:
        messages.insert(0, notice)
    return messages


def _update_tool_result_cache(
    cache: Optional[Dict[str, str]], results: List[Dict[str, Any]]
) -> Dict[str, str]:
    """Records tool results keyed by ``tool_use_id``, bounded to recent entries.

    Memoization exists so a retried ``tool_calls`` request can be answered from
    the cache instead of re-running side effects. Storing only the id (as
    ``processed_tool_use_ids`` does) is not enough: the host is told a call was
    processed but is given no way to recover what it produced, so the safe
    response is to skip the call and lose the result.

    The cache is capped because it is persisted in the run state; without a bound
    it would simply relocate the unbounded-growth problem this work removed.
    Re-recording an id moves it to the newest slot so eviction keeps live calls.
    """
    merged: Dict[str, str] = dict(cache or {})
    for r in results or []:
        if not isinstance(r, dict):
            continue
        tid = r.get("tool_use_id")
        if not isinstance(tid, str) or not tid:
            continue
        content = r.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = Yaml.dump_to_string(content).strip()
        merged.pop(tid, None)
        merged[tid] = text
    if len(merged) > TOOL_RESULT_CACHE_CAP:
        for old in list(merged)[: len(merged) - TOOL_RESULT_CACHE_CAP]:
            merged.pop(old, None)
    return merged


def write_llm_request(
        history: List[Dict[str, Any]], rnd: int, tool_depth: int, min_rounds: int,
        retry: bool = False, applied_lessons: Optional[List[str]] = None,
        min_override: Optional[int] = None, rounds: Optional[Dict[str, Any]] = None
    ) -> None:
    """Writes the current prompt/history state and requests LLM generation.

    Only bounded history windows are sent and persisted. The full transcript is
    not written to disk, because nothing ever reads it back: the request uses the
    send window, exit and deadlock detection read the score history in
    ``JINX.yaml``, and a retry only needs the most recent tool calls.

    When messages fall outside the window a one-line notice is prepended to the
    request only. It is deliberately not persisted, so the synthetic note cannot
    accumulate one entry per round.

    ``min_override`` is the ``--min`` value from the command line. Each round is a
    new process, so the override only lasts if it is written to the run state;
    None keeps whatever the run state already holds.

    ``rounds`` is the adaptive round ledger (see ``_ledger_of``). It is rebuilt
    from an explicit key list like everything else here, so a caller that has
    updated the ledger in memory has to hand it over; None keeps the stored one.
    """
    carried: Dict[str, Any] = {}
    try:
        if RUN_STATE_PATH.exists():
            existing = Yaml.load_from_file(RUN_STATE_PATH)
            if isinstance(existing, dict):
                carried = existing
    except Exception:
        carried = {}

    processed_ids: List[str] = carried.get("processed_tool_use_ids", []) or []
    result_cache: Dict[str, str] = carried.get("tool_result_cache", {}) or {}
    override = min_override if min_override is not None else carried.get("min_override")

    # The notice describes what the MODEL was not shown, so it is measured against
    # the send window; the persist window is deliberately larger.
    messages = _request_messages(history)

    # A self-patch reverted by the bootstrap preflight leaves its explanation in
    # the run state. It is surfaced here because this is the one funnel every
    # llm_generate request passes through, and the run state is rebuilt from
    # explicit keys below, so the marker is consumed exactly once.
    feedback = carried.get("self_patch_feedback")
    if isinstance(feedback, str) and feedback:
        messages.insert(0, {"role": "user", "content": feedback})

    request_payload = {
        "type": "llm_generate", "system": SYSTEM_PROMPT,
        "messages": messages, "tools": tool_schema(),
        "processed_tool_use_ids": processed_ids,
        "tool_result_cache": result_cache, "retry": bool(retry)
    }
    try:
        Yaml.safe_atomic_write(REQUEST_PATH, request_payload)
    except JinxError as e:
        raise IPCError(f"Failed to write request: {e}") from e

    run_state = {
        "rnd": rnd, "tool_depth": tool_depth, "history": _persist_window(history),
        "waiting_for": "llm_generate", "min_rounds": min_rounds,
        "processed_tool_use_ids": processed_ids,
        "tool_result_cache": result_cache,
        # Carried, not recomputed: this function rebuilds the run state from an
        # explicit key list, so anything a caller stashed in the in-memory dict
        # (such as the lessons shown this round) would otherwise be silently
        # dropped on every persist.
        "applied_lessons": (
            applied_lessons if applied_lessons is not None
            else carried.get("applied_lessons") or []
        ),
        "updated_at": time.time()
    }
    if override is not None:
        run_state["min_override"] = override
    ledger = rounds if rounds is not None else carried.get("rounds")
    if isinstance(ledger, dict):
        run_state["rounds"] = ledger
    try:
        Yaml.safe_atomic_write(RUN_STATE_PATH, run_state)
    except JinxError as e:
        raise IPCError(f"Failed to write run state: {e}") from e

    print(f"[JINX_WAITING] Requesting LLM completion for Round {rnd}...", flush=True)


def run_file_ipc(task: Optional[str], min_override: Optional[int]) -> None:
    """Orchestrates JINX loop using a stateless File-based IPC protocol."""
    _install_signal_handlers()
    # Claimed here rather than at each write site: the IPC files are written by
    # four different helpers (write_llm_request, _write_tool_request,
    # _write_llm_request_no_tools and _handle_tool_response), and a run that
    # goes through the tool-call or no-tools paths would otherwise create the
    # files without owning them, leaving nothing to clean up on Ctrl+C. Starting
    # a file-IPC session is exactly what makes this process the owner, and it is
    # the one thing a bystander process never does.
    _claim_ipc_files()
    is_resuming = RUN_STATE_PATH.exists() and not task

    if not is_resuming:
        if not task:
            logger.error("Cannot start new session without a task description.")
            sys.exit(1)

        jinx = read_jinx()
        _init_new_session(task, jinx)
        min_rounds = _resolve_min_rounds(jinx, min_override)
        clean_up_ipc_files()

        # Baseline the framework source before any tool call this round can
        # modify it, and start from a fresh lesson-credit slate.
        if selfpatch.SELF_PATCH_ENABLED:
            selfpatch.capture_baseline()

        run_state: Dict[str, Any] = {
            "rnd": 1, "tool_depth": 0, "history": [],
            "waiting_for": "llm_generate", "min_rounds": min_rounds,
        }
        # The count shown on the first label is what a later "raised from X" is
        # measured against, so it is recorded with the ledger from the start.
        ledger = _ledger_of(run_state)
        ledger["shown"] = min_rounds
        lessons_text, applied = _inject_lessons(run_state)

        state_dump = Yaml.dump_to_string(jinx["state"])
        user_msg = construct_round_prompt(
            rnd=1, min_rounds=min_rounds, state_dump=state_dump,
            lessons_text=lessons_text,
        )
        try:
            write_llm_request(
                [{"role": "user", "content": user_msg}], 1, 0, min_rounds,
                applied_lessons=applied, min_override=min_override, rounds=ledger,
            )
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

    required_keys = ("rnd", "tool_depth", "history", "waiting_for", "min_rounds")
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

    if not RESPONSE_PATH.exists():
        if _is_run_state_stale(run_state):
            logger.warning(
                "Waiting for %s timed out after %ss; reissuing request in background-safe retry mode.",
                waiting_for, BACKGROUND_WAIT_TIMEOUT
            )
            if waiting_for == "tool_calls":
                tool_calls = _extract_last_tool_calls(history)
                if tool_calls:
                    _write_tool_request(tool_calls, history, rnd, tool_depth, min_rounds, run_state, retry=True)
                    return
            write_llm_request(history, rnd, tool_depth, min_rounds, retry=True)
            return
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
                _handle_llm_response(response_data, history, rnd, tool_depth, min_rounds, run_state)
            except (IPCError, OSError, JinxError) as e:
                logger.error("IPC failure while handling LLM response: %s", e, exc_info=True)
                clean_up_ipc_files()
                sys.exit(1)
        elif waiting_for == "tool_calls":
            try:
                _handle_tool_response(response_data, history, rnd, tool_depth, min_rounds, run_state)
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
    rnd: int, tool_depth: int, min_rounds: int,
    run_state: Dict[str, Any]
) -> None:
    """Handles the response from an LLM generation request."""
    raw_content = response_data.get("content")
    if isinstance(raw_content, str):
        content_blocks = [{"type": "text", "text": raw_content}]
    elif isinstance(raw_content, list):
        content_blocks = [
            b if isinstance(b, dict) else {"type": "text", "text": str(b)}
            for b in raw_content
        ]
    elif raw_content is not None:
        content_blocks = [{"type": "text", "text": str(raw_content)}]
    else:
        content_blocks = [{"type": "text", "text": ""}]

    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    history.append({"role": "assistant", "content": content_blocks})

    full_text = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )

    tool_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
    valid_calls: List[Dict[str, Any]] = []
    malformed_results: List[Dict[str, Any]] = []
    refused_results: List[Dict[str, Any]] = []

    for b in tool_blocks:
        tool_use_id, name, params, err = _validate_tool_use_block(b)
        if err:
            malformed_results.append(err)
            continue
        # Refuse a self-patch that would rewrite brake logic BEFORE dispatching
        # it. Undo-after-the-fact is not good enough here: the model must be told
        # it was refused, or it will keep re-attempting the same edit and read
        # the failure as "verification is broken".
        if name == "file_write" and isinstance(params, dict):
            reason = selfpatch.guard_tool_call(
                str(params.get("path") or ""), str(params.get("content") or "")
            )
            if reason:
                logger.warning("Refused protected self-patch: %s", params.get("path"))
                refused_results.append(
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": reason}
                )
                continue
        valid_calls.append({"id": tool_use_id, "name": name, "params": params})

    if malformed_results:
        history.append({"role": "user", "content": malformed_results})

    if refused_results:
        # Paired refusals are delivered as a normal tool_result turn so the model
        # stays in the tool-calling protocol instead of falling through to a
        # state-block parse it did not intend.
        history.append({"role": "user", "content": refused_results})
        if not valid_calls:
            # Every call was refused, but the loop still advanced: the model spent
            # a turn. Counting it against the depth cap is what stops a model that
            # keeps re-proposing the same forbidden write from spinning forever
            # without ever reaching the cap that would break it out.
            next_depth = tool_depth + 1
            if next_depth >= TOOL_DEPTH_CAP:
                logger.warning(
                    "Tool depth limit reached while refusing self-patches. "
                    "Forcing state recovery."
                )
                run_state["tool_depth"] = next_depth
                history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": TOOL_DEPTH_CRITICAL_MSG}],
                })
                try:
                    _write_llm_request_no_tools(history, rnd, run_state)
                except (IPCError, OSError, JinxError) as e:
                    logger.error(
                        "IPC failure after refusing a self-patch: %s", e, exc_info=True
                    )
                    clean_up_ipc_files()
                    sys.exit(1)
                return
            try:
                write_llm_request(history, rnd, next_depth, min_rounds)
            except (IPCError, OSError, JinxError) as e:
                logger.error("IPC failure after refusing a self-patch: %s", e, exc_info=True)
                clean_up_ipc_files()
                sys.exit(1)
            return

    if valid_calls:
        try:
            _write_tool_request(valid_calls, history, rnd, tool_depth + 1, min_rounds, run_state)
        except (IPCError, OSError, JinxError) as e:
            logger.error("IPC failure while writing tool_calls request: %s", e, exc_info=True)
            clean_up_ipc_files()
            sys.exit(1)
        return

    # No tool calls — parse state block
    update = parse_state_block(full_text)
    jinx = read_jinx()
    diagnostics: List[str] = []
    # The round count is recomputed every round, from the sized count and what the
    # runner saw this round (see _close_round). A --min override is kept in the run
    # state and still pins the sized count. Even a reply with no state block goes
    # through this: a round that ran nothing and recorded nothing is a dead round.
    min_override = run_state.get("min_override")
    round_ledger = _ledger_of(run_state)
    closed = _close_round(jinx, update, round_ledger, min_override, rnd, diagnostics)
    jinx, scores = closed["jinx"], closed["scores"]
    floor, min_rounds = closed["floor"], closed["exit_floor"]
    if update:
        outcome: Dict[str, Any] = closed["outcome"]

        # Honour exit/deadlock ONLY from state that actually passed validation. A
        # rejected block must not be able to terminate the loop or claim success
        # on the strength of scores the run state never accepted.
        if outcome.get("applied"):
            # Persist any NEW durable lessons to the cross-run ledger, and settle
            # the credit for the lessons this round was actually shown. Both are
            # best-effort: the learning store must never fail a round.
            incoming = outcome.get("lessons") or []
            if incoming:
                try:
                    ledger = learning.load_ledger()
                    ledger["lessons"] = learning.add_lessons(
                        ledger.get("lessons"), incoming
                    )
                    learning.save_ledger(ledger)
                except Exception as e:
                    logger.error("Could not persist lessons: %s", e, exc_info=True)
            _close_lesson_bookkeeping(
                run_state, bool(jinx["state"].get("scores", [{}])[-1].get("all_pass")
                                if jinx["state"].get("scores") else False)
            )

            verdict = _round_verdict(
                jinx, scores, round_ledger, closed["dead"], floor, min_rounds, rnd,
                diagnostics,
            )
            if verdict == "exit":
                print("[JINX_COMPLETE] Task resolved successfully!", flush=True)
                _finish_run()
                return

            if verdict == "deadlock":
                print("[JINX_DEADLOCK] Loop aborted due to strategy deadlock.", flush=True)
                _finish_run()
                return
        else:
            logger.warning(
                "State block rejected; ignoring exit_ready/deadlock flags from the "
                "rejected response and continuing to the next round."
            )

    # Transition to next round
    rnd += 1
    if rnd >= HARD_CAP:
        _finish_run()
        logger.error("Cognitive loop exhausted HARD_CAP.")
        sys.exit(2)

    jinx = read_jinx()
    state_dump = Yaml.dump_to_string(jinx.get("state") or {})
    lessons_text, applied = _inject_lessons(run_state)
    adaptive = _adaptive_info(
        jinx, closed["base"], round_ledger, floor, min_rounds, closed["dead"], rnd,
        min_override,
    )
    user_msg = construct_round_prompt(
        rnd=rnd, min_rounds=min_rounds, state_dump=state_dump,
        missing_state=not update, lessons_text=lessons_text, adaptive=adaptive,
    )
    round_ledger["shown"] = min_rounds
    if diagnostics:
        user_msg = user_msg + "\n" + "\n".join(diagnostics) + "\n"
    history.append({"role": "user", "content": user_msg})
    try:
        write_llm_request(
            history, rnd, 0, min_rounds, applied_lessons=applied, rounds=round_ledger,
        )
    except (IPCError, OSError, JinxError) as e:
        logger.error("IPC failure while writing next LLM request: %s", e, exc_info=True)
        clean_up_ipc_files()
        sys.exit(1)


def _handle_tool_response(
    response_data: Dict[str, Any], history: List[Dict[str, Any]],
    rnd: int, tool_depth: int, min_rounds: int,
    run_state: Dict[str, Any]
) -> None:
    """Handles the response from tool execution."""
    results = response_data.get("results") or []
    tool_results = [
        {"type": "tool_result", "tool_use_id": r.get("tool_use_id"), "content": r.get("content") or ""}
        for r in results
    ]

    # The model may edit JINX's own source. That is allowed and sometimes the
    # right move, but it must not be allowed to leave a broken framework behind,
    # so verify-and-revert runs before the results are handed back for a new turn.
    feedback = _enforce_self_patch_gate(run_state)
    if feedback:
        # Appended after the tool results, not before: the protocol requires every
        # tool_result first, and the model should read the verdict as a
        # conclusion on those results rather than as an instruction preceding them.
        tool_results.append({"type": "text", "text": feedback})

    # Measure what the batch returned against what the same calls returned before.
    # A call that keeps returning the identical result cannot change anything, and
    # the model is told so now, in the results it is about to read, rather than in
    # the next round's prompt after the loop of identical calls has been spent.
    round_ledger = _ledger_of(run_state)
    repeats = _record_tool_results(round_ledger, _tool_calls_in(history), results)
    if repeats:
        tool_results.append({"type": "text", "text": prompts.note_repeated_call(repeats)})

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

    # Persist processed ids AND the results themselves. Ids alone tell a host that
    # a call ran but not what it produced, so a retry could only skip it and lose
    # the output; with the cache the retry can be answered from memory.
    try:
        processed = run_state.get("processed_tool_use_ids", []) or []
        for r in results:
            tid = r.get("tool_use_id")
            if isinstance(tid, str) and tid not in processed:
                processed.append(tid)
        run_state["processed_tool_use_ids"] = processed
        run_state["tool_result_cache"] = _update_tool_result_cache(
            run_state.get("tool_result_cache"), results
        )
        try:
            Yaml.safe_atomic_write(RUN_STATE_PATH, run_state)
        except JinxError:
            logger.warning("Failed to persist tool result cache to run state.")
    except Exception:
        logger.debug("Unable to update tool result cache in run_state.", exc_info=True)

    try:
        write_llm_request(history, rnd, tool_depth, min_rounds, rounds=round_ledger)
    except (IPCError, OSError, JinxError) as e:
        logger.error("IPC failure while writing LLM request after tool response: %s", e, exc_info=True)
        clean_up_ipc_files()
        sys.exit(1)


def _write_tool_request(
    tool_calls: List[Dict[str, Any]], history: List[Dict[str, Any]],
    rnd: int, tool_depth: int, min_rounds: int, run_state: Dict[str, Any],
    retry: bool = False
) -> None:
    """Writes a tool_calls request and updates run state."""
    processed_ids: List[str] = run_state.get("processed_tool_use_ids", []) or []
    result_cache: Dict[str, str] = _update_tool_result_cache(
        run_state.get("tool_result_cache"), []
    )

    request_payload = {
        "type": "tool_calls", "calls": tool_calls,
        "processed_tool_use_ids": processed_ids,
        "tool_result_cache": result_cache, "retry": bool(retry)
    }
    try:
        Yaml.safe_atomic_write(REQUEST_PATH, request_payload)
    except JinxError as e:
        # Propagate as IPCError so callers can clean up IPC files.
        raise IPCError(f"Failed to write tool_calls request: {e}") from e

    run_state.update({
        "tool_depth": tool_depth,
        "history": _persist_window(history),
        "waiting_for": "tool_calls", "updated_at": time.time(),
        "tool_result_cache": result_cache,
    })
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
        "type": "llm_generate", "system": SYSTEM_PROMPT,
        "messages": _request_messages(history), "tools": []
    }
    try:
        Yaml.safe_atomic_write(REQUEST_PATH, request_payload)
    except JinxError as e:
        raise IPCError(f"Failed to write final summary request: {e}") from e

    run_state.update({
        "waiting_for": "llm_generate",
        "history": _persist_window(history),
        "updated_at": time.time()
    })
    try:
        Yaml.safe_atomic_write(RUN_STATE_PATH, run_state)
    except JinxError as e:
        raise IPCError(f"Failed to write run state: {e}") from e

    print(f"[JINX_WAITING] Requesting final summary for Round {rnd}...", flush=True)


def run(task: Optional[str], min_override: Optional[int], ipc_mode: str = "file") -> None:
    """Orchestrates the JINX execution loop."""
    _install_signal_handlers()
    if ipc_mode == "file":
        run_file_ipc(task, min_override)
        return

    # Interactive duplex stream JSON-RPC mode
    jinx = read_jinx()
    _init_new_session(task or "", jinx)
    min_rounds = _resolve_min_rounds(jinx, min_override)

    rnd: int = 0
    last_round_missing_state: bool = False
    history: List[Dict[str, Any]] = []
    # The same adaptive round count as the file-IPC path. This mode is one process,
    # so the ledger lives in memory instead of the run state, and the count is
    # re-derived every round: it used to be resolved once, before the loop, so a
    # protocol change from the model never took effect here.
    round_ledger = _new_ledger()
    adaptive: Optional[Dict[str, Any]] = None
    pending_notes: List[str] = []

    logger.info("Starting JINX loop (JSON-RPC). Task: '%s'. Min rounds: %d", task, min_rounds)

    while rnd < HARD_CAP:
        rnd += 1
        jinx = read_jinx()
        state_data = jinx.get("state") or {}
        state_dump = Yaml.dump_to_string(state_data)

        user_msg = construct_round_prompt(
            rnd=rnd, min_rounds=min_rounds,
            state_dump=state_dump, missing_state=last_round_missing_state,
            adaptive=adaptive,
        )
        round_ledger["shown"] = min_rounds
        if pending_notes:
            user_msg = user_msg + "\n" + "\n".join(pending_notes) + "\n"
            pending_notes = []
        history.append({"role": "user", "content": user_msg})

        full_text: str = ""
        tool_depth: int = 0
        try:
            while True:
                content_blocks = request_llm_from_editor(SYSTEM_PROMPT, history)
                history.append({"role": "assistant", "content": content_blocks})

                for block in content_blocks:
                    if block.get("type") == "text":
                        full_text += block.get("text", "")

                tool_results: List[Dict[str, Any]] = []
                calls_here: Dict[str, Dict[str, Any]] = {}
                for block in content_blocks:
                    if block.get("type") == "tool_use":
                        tool_results.append(_execute_rpc_tool(block))
                        call_id, call_name, call_params, call_err = _validate_tool_use_block(block)
                        if call_err is None and call_id is not None:
                            calls_here[call_id] = {
                                "id": call_id, "name": call_name, "params": call_params,
                            }

                if tool_results:
                    repeats = _record_tool_results(round_ledger, calls_here, tool_results)
                    if repeats:
                        tool_results.append(
                            {"type": "text", "text": prompts.note_repeated_call(repeats)}
                        )
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
        except IPCError as e:
            logger.critical("IPC failure during RPC loop: %s", e, exc_info=True)
            clean_up_ipc_files()
            sys.exit(1)

        update = parse_state_block(full_text)
        round_notes: List[str] = []
        closed = _close_round(jinx, update, round_ledger, min_override, rnd, round_notes)
        jinx = closed["jinx"]
        floor, min_rounds = closed["floor"], closed["exit_floor"]
        if update:
            last_round_missing_state = False

            # Same rule as the File-IPC path: a rejected block cannot terminate
            # the loop. Its flags are ignored and the round continues.
            if not closed["applied"]:
                logger.warning("State block rejected in RPC mode; ignoring flags.")
            else:
                verdict = _round_verdict(
                    jinx, closed["scores"], round_ledger, closed["dead"], floor,
                    min_rounds, rnd, round_notes,
                )
                if verdict == "exit":
                    logger.info("Execution complete in round %d.", rnd)
                    break
                if verdict == "deadlock":
                    logger.warning("Deadlock in round %d.", rnd)
                    break
        else:
            last_round_missing_state = True
        adaptive = _adaptive_info(
            jinx, closed["base"], round_ledger, floor, min_rounds, closed["dead"],
            rnd + 1, min_override,
        )
        pending_notes = round_notes
    else:
        logger.error("HARD_CAP (%d rounds) exhausted.", HARD_CAP)
        if selfpatch.SELF_PATCH_ENABLED:
            selfpatch.clear_baseline()
        sys.exit(2)


def _execute_rpc_tool(block: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a single tool call in RPC mode."""
    tool_use_id, name, params, err = _validate_tool_use_block(block)
    if err:
        return err
    # Same brake-removal guard as the File-IPC path: a refused self-patch is
    # never dispatched to the editor.
    if name == "file_write" and isinstance(params, dict):
        reason = selfpatch.guard_tool_call(
            str(params.get("path") or ""), str(params.get("content") or "")
        )
        if reason:
            logger.warning("Refused protected self-patch in RPC mode: %s", params.get("path"))
            return {"type": "tool_result", "tool_use_id": tool_use_id, "content": reason}
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
        return prompts.FILE_SLICE_FAILURE_MSG % e


_STDIN_QUEUE: "_queue.Queue[Optional[str]]" = _queue.Queue()
_STDIN_THREAD: Optional[threading.Thread] = None
_STDIN_LOCK = threading.Lock()
def _ensure_stdin_reader() -> None:
    """Starts the single background stdin reader once per process."""
    global _STDIN_THREAD
    with _STDIN_LOCK:
        if _STDIN_THREAD is not None and _STDIN_THREAD.is_alive():
            return
        def _reader() -> None:
            while True:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    _STDIN_QUEUE.put(None)
                    return
                if line == "":
                    _STDIN_QUEUE.put(None)
                    return
                _STDIN_QUEUE.put(line)
        _STDIN_THREAD = threading.Thread(target=_reader, daemon=True)
        _STDIN_THREAD.start()
def _read_stdin_line(timeout: int) -> Optional[str]:
    """Reads a line from the shared stdin queue with a timeout."""
    _ensure_stdin_reader()
    try:
        return _STDIN_QUEUE.get(timeout=timeout)
    except _queue.Empty:
        return None
def _read_stdin_with_retries(timeout: int, retries: int, backoff: float) -> Optional[str]:
    """Attempt to read a line from stdin multiple times with backoff.
    Returns the line or None only after exhausting retries.
    """
    attempt = 0
    while attempt < retries:
        line = _read_stdin_line(timeout)
        if line is not None:
            return line
        attempt += 1
        if attempt < retries:
            try:
                time.sleep(backoff)
            except Exception:
                pass
    return None
