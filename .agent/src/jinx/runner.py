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
from .state import merge_state, read_jinx, write_jinx
from .tools import tool_schema
from . import learning, selfpatch

logger = logging.getLogger("jinx.runner")

HARD_CAP: int = 40
TOOL_DEPTH_CAP: int = 20

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
        try:
            clean_up_ipc_files()
        except Exception:
            pass
    # Use os._exit to avoid sys.exit raising SystemExit inside a signal handler,
    # which can cause recursion if the handler itself was invoked during cleanup.
    try:
        os._exit(1)
    except Exception:
        pass


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
    # NOTE: the durable lesson ledger in .agent/lessons.yaml is deliberately left
    # untouched. A new task wipes working memory, but what was learned is exactly
    # what should carry over; resetting it here would make the store decay at the
    # same rate as the state it is supposed to outlast.


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
        return (
            "SELF-PATCH REFUSED and ROLLBACK FAILED: %s\nThe rollback itself "
            "raised %s: %s. Treat JINX's source as untrustworthy and ask a human "
            "before continuing — the next run's preflight will retry the repair "
            "from the baseline." % (reason, type(e).__name__, e)
        )
    logger.warning("Self-patch protection gate reverted %d file(s)", len(restored))
    return (
        "SELF-PATCH REFUSED: %s\nFiles rolled back: %s"
        % (reason, ", ".join(restored) or "none")
    )


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
        selfpatch.restore_baseline(only=selfpatch.REPO_PREFIX)
    except Exception as e:
        logger.error("Could not restore the test files: %s", e, exc_info=True)
        return _rollback_and_report(
            "the test files that decide whether this patch is acceptable could "
            "not be restored (%s: %s), so it was not verified against a trusted "
            "suite" % (type(e).__name__, e)
        )

    result = selfpatch.verify(AGENT_DIR.parent)
    if result["ok"]:
        logger.info("Self-patch verified: %s", ", ".join(changed))
        # Adopt the verified edit as the new reference so the same diff is not
        # re-verified on every later round.
        selfpatch.capture_baseline()
        return None

    restored = selfpatch.restore_baseline()
    message = (
        "SELF-PATCH REVERTED: your edit to the files JINX verifies (%s) failed "
        "verification (%s). The change was rolled back automatically, so the "
        "framework is intact — the round was not wasted, it produced evidence. "
        "Read the failure below, decide whether the idea is still right, and if "
        "so apply it in a smaller or more targeted form.\n%s"
        % (
            ", ".join(changed),
            result["summary"],
            "\n".join("[%s] %s" % (c["name"], c["tail"]) for c in result["checks"]),
        )
    )
    logger.warning("Self-patch gate reverted %d file(s)", len(restored))
    return message


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
    """Extracts the latest tool_use blocks from the accumulated history."""
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
            if block.get("type") == "tool_use":
                tool_use_id, name, params, err = _validate_tool_use_block(block)
                if err is None and tool_use_id is not None and name is not None:
                    calls.append({"id": tool_use_id, "name": name, "params": params})
        if calls:
            return calls
    return []


HISTORY_WINDOW: int = int(os.getenv("JINX_HISTORY_WINDOW", "6"))
HISTORY_PERSIST_WINDOW: int = int(os.getenv("JINX_HISTORY_PERSIST_WINDOW", "8"))
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
    never receives a result whose request is not in the window. This bounds both
    what is sent and what is persisted, which is what keeps the run-state file
    from growing without limit across a long task.
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
        "content": (
            "[context note] %d earlier message(s) from this session were elided from "
            "the history window to bound prompt size; %d of them involved tool "
            "traffic. Their substance is preserved in the score history in CURRENT "
            "STATE (see 'scores'). Do not assume this window is the whole session."
            % (len(rounds), tool_msgs)
        ),
    }


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
        retry: bool = False, applied_lessons: Optional[List[str]] = None
    ) -> None:
    """Writes the current prompt/history state and requests LLM generation.

    Only the bounded history window is both sent and persisted. The full
    transcript is not written to disk, because nothing ever reads it back: the
    request uses the window, exit and deadlock detection read the score history
    in ``JINX.yaml``, and tool dispatch only needs the most recent calls.

    When messages fall outside the window a one-line notice is prepended to the
    request only. It is deliberately not persisted, so the synthetic note cannot
    accumulate one entry per round.
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

    persist_window = compact_history_for_request(history, HISTORY_PERSIST_WINDOW)
    send_window = compact_history_for_request(history)

    # The notice describes what the MODEL was not shown, so the baseline must be
    # the send window. The persist window is deliberately larger, so measuring
    # against it would silently omit the messages that live in the gap between the
    # two and under-report the elided count on every round.
    kept = {id(m) for m in send_window}
    dropped = [m for m in history if id(m) not in kept]
    messages = list(send_window)
    notice = summarize_dropped_history(dropped, send_window)
    if notice:
        messages.insert(0, notice)

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
        "rnd": rnd, "tool_depth": tool_depth, "history": persist_window,
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

        # Baseline the framework source before any tool call this round can
        # modify it, and start from a fresh lesson-credit slate.
        if selfpatch.SELF_PATCH_ENABLED:
            selfpatch.capture_baseline()

        run_state: Dict[str, Any] = {
            "rnd": 1, "tool_depth": 0, "history": [],
            "waiting_for": "llm_generate", "min_rounds": min_rounds,
        }
        lessons_text, applied = _inject_lessons(run_state)

        state_dump = Yaml.dump_to_string(jinx["state"])
        user_msg = construct_round_prompt(
            rnd=1, min_rounds=min_rounds, state_dump=state_dump,
            lessons_text=lessons_text,
        )
        try:
            write_llm_request(
                [{"role": "user", "content": user_msg}], 1, 0, min_rounds,
                applied_lessons=applied,
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
    if update:
        outcome: Dict[str, Any] = {}
        jinx = merge_state(jinx, update, diagnostics=diagnostics, outcome=outcome)
        write_jinx(jinx)
        # Re-resolve min_rounds so protocol changes from LLM take effect mid-session
        min_rounds = _resolve_min_rounds(jinx, None)
        scores = jinx["state"].get("scores", [])

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

            flags = jinx["state"]
            if flags.get("exit_ready") and check_exit(scores, min_rounds, rnd):
                print("[JINX_COMPLETE] Task resolved successfully!", flush=True)
                _finish_run()
                return

            if flags.get("deadlock") or check_deadlock(scores, min_rounds, rnd):
                if not flags.get("deadlock"):
                    jinx["state"]["deadlock"] = True
                    write_jinx(jinx)
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
    user_msg = construct_round_prompt(
        rnd=rnd, min_rounds=min_rounds, state_dump=state_dump,
        missing_state=not update, lessons_text=lessons_text,
    )
    if diagnostics:
        user_msg = user_msg + "\n" + "\n".join(diagnostics) + "\n"
    history.append({"role": "user", "content": user_msg})
    try:
        write_llm_request(history, rnd, 0, min_rounds, applied_lessons=applied)
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
        write_llm_request(history, rnd, tool_depth, min_rounds)
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
        "history": compact_history_for_request(history, HISTORY_PERSIST_WINDOW),
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
        "messages": compact_history_for_request(history), "tools": []
    }
    try:
        Yaml.safe_atomic_write(REQUEST_PATH, request_payload)
    except JinxError as e:
        raise IPCError(f"Failed to write final summary request: {e}") from e

    run_state.update({
        "waiting_for": "llm_generate",
        "history": compact_history_for_request(history, HISTORY_PERSIST_WINDOW),
        "updated_at": time.time()
    })
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
    history: List[Dict[str, Any]] = []

    logger.info("Starting JINX loop (JSON-RPC). Task: '%s'. Min rounds: %d", task, min_rounds)

    while rnd < HARD_CAP:
        rnd += 1
        jinx = read_jinx()
        state_data = jinx.get("state") or {}
        state_dump = Yaml.dump_to_string(state_data)

        user_msg = construct_round_prompt(
            rnd=rnd, min_rounds=min_rounds,
            state_dump=state_dump, missing_state=last_round_missing_state
        )
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
        except IPCError as e:
            logger.critical("IPC failure during RPC loop: %s", e, exc_info=True)
            clean_up_ipc_files()
            sys.exit(1)

        update = parse_state_block(full_text)
        if update:
            last_round_missing_state = False
            outcome: Dict[str, Any] = {}
            jinx = merge_state(jinx, update, outcome=outcome)
            write_jinx(jinx)
            scores = jinx["state"].get("scores", [])

            # Same rule as the File-IPC path: a rejected block cannot terminate
            # the loop. Its flags are ignored and the round continues.
            if not outcome.get("applied"):
                logger.warning("State block rejected in RPC mode; ignoring flags.")

            if outcome.get("applied") and jinx["state"].get("exit_ready") \
                    and check_exit(scores, min_rounds, rnd):
                logger.info("Execution complete in round %d.", rnd)
                break
            if outcome.get("applied") and (jinx["state"].get("deadlock")
                                           or check_deadlock(scores, min_rounds, rnd)):
                logger.warning("Deadlock in round %d.", rnd)
                if not jinx["state"].get("deadlock"):
                    jinx["state"]["deadlock"] = True
                    write_jinx(jinx)
                break
        else:
            last_round_missing_state = True
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
        return f"Error: Failed to slice file content: {e}"


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
