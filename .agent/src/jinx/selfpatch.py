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
"""Gated self-patching — let JINX edit its own code without letting it remove
its own brakes.

The problem this solves
-----------------------
``file_write`` accepts a bare string path and the shipped reference host calls
``Path(params["path"]).write_text(...)`` with no restriction at all. The model
has therefore *always* been able to rewrite ``.agent/src/jinx/*.py`` — and has,
which is how this repository's own release work happens. Nothing in the codebase
checks whether the result still works, and nothing stops it from disabling the
very logic that would have noticed.

So the gate is not a proposal queue; the model can already write. The gate is
**verification and rollback**:

1. Snapshot ``.agent/src`` before the round.
2. If the round wrote to the framework's own source, compile it and import every module in a child
   interpreter, to reject syntax errors and source that cannot be loaded.
3. If either check fails, restore the snapshot and report the failure.

Autonomous self-modification is only defensible with a brake-removal guard on
top, because otherwise "the agent edited its own guardrails" and "the agent is
self-improving" become indistinguishable. See :data:`PROTECTED_SYMBOLS`.
"""

import contextlib
import errno
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import prompts

logger = logging.getLogger("jinx.selfpatch")

# The source tree JINX is allowed to self-patch.
SRC_DIR: Path = Path(
    os.environ.get("JINX_SRC_DIR", str(Path(__file__).resolve().parent))
)

# Set to 0 to disable the gate entirely (protection and rollback).
SELF_PATCH_ENABLED: bool = os.environ.get("JINX_SELF_PATCH", "1") not in (
    "0",
    "false",
    "False",
)

# Set to 1 to allow edits that touch protected symbols. Off by default: this is
# the escape hatch for a human maintainer, not for the agent.
ALLOW_PROTECTED: bool = os.environ.get("JINX_ALLOW_PROTECTED_EDITS", "0") in (
    "1",
    "true",
    "True",
)

VERIFY_TIMEOUT: int = int(os.environ.get("JINX_SELF_PATCH_TIMEOUT", "600"))

# Files that may never be self-patched. ``selfpatch.py`` and ``learning.py`` are
# included because otherwise the agent could disable the gate or the credit
# assignment that makes the ledger meaningful.
PROTECTED_FILES = ("selfpatch.py", "learning.py")

# Brake logic, protected at symbol granularity so the rest of the same file can
# still be improved. Each entry is (filename, regex) and matches a definition
# line, so renaming a function breaks the match and re-opens the brake — that is
# deliberately conservative in the safe direction.
PROTECTED_SYMBOLS: Tuple[Tuple[str, str], ...] = (
    # The whole-block validation and its "applied" trust gate.
    ("state.py", r"^def merge_state"),
    ("state.py", r"^class StateBlock"),
    ("state.py", r"^def _resolve_jinx_path"),
    ("state.py", r"^def atomic_write_yaml"),
    # How the model's protocol section is merged, and how its sizing is marked.
    ("state.py", r"^def _apply_protocol"),
    # Exit / deadlock policy and the flags gate in both transports.
    ("runner.py", r"^def check_exit"),
    ("runner.py", r"^def check_deadlock"),
    ("runner.py", r"^def _resolve_min_rounds"),
    ("runner.py", r"^def _handle_llm_response"),
    # The adaptive round count: what may only ever raise it, what charges a round
    # that did nothing, and what refuses an early stop. An agent that could edit
    # these could make the loop shorter, which is the one thing they exist to stop.
    ("runner.py", r"^def (_ledger_of|_finish_round|_note_round_result|_discard_sizing)\b"),
    ("runner.py", r"^def (_scores_of|_loop_of)\b"),
    ("runner.py", r"^def (_declared_base|_store_declared|_round_floors)\b"),
    ("runner.py", r"^def (_close_round|_round_verdict|_record_tool_results)\b"),
    ("runner.py", r"^(DEFAULT_MIN_ROUNDS|SIZING_ROUNDS|CONFIRM_ROUNDS|DEAD_ROUND_CAP)\b"),
    # The prompt contract itself.
    ("prompts.py", r"^SYSTEM_PROMPT"),
)

class ProtectionError(RuntimeError):
    """Raised when a self-patch would modify protected brake logic."""


def _protected_blocks(text: str, pattern: str) -> Tuple[str, ...]:
    """Extracts every top-level unit a protected pattern starts.

    The unit runs from the matched line to the next line that begins at column
    zero with a non-comment character, which is where the next top-level
    definition or assignment starts. Indented continuation lines and comments
    stay inside the unit, so a function cannot be quietly truncated.

    Every match is collected, not just the first. Python keeps the *last*
    definition of a name, so appending a second `def merge_state` to the end of
    the file silently replaces the validated one at import time. Looking only at
    the first match would see an untouched block and wave that through, which is
    the exact bypass the guard exists to prevent.
    """
    blocks: List[str] = []
    for match in re.finditer(pattern, text, re.MULTILINE):
        rest = text[match.start():]
        end = re.search(r"\n(?=[^\s#])", rest)
        block = rest[:end.start()] if end else rest
        # Drop trailing blank lines and comment-only lines. Without this, adding
        # an unrelated function or a note *after* a protected definition changed
        # that definition's extracted text, so an ordinary improvement elsewhere
        # in the file was reported as an attack on the brake. Neither trailing
        # blank lines nor trailing comments can change behaviour.
        body = block.splitlines()
        while body and (not body[-1].strip() or body[-1].lstrip().startswith("#")):
            body.pop()
        blocks.append("\n".join(body))
    return tuple(blocks)


def _read_existing(relative_path: Path) -> Optional[str]:
    """Reads the current on-disk source a candidate edit would replace.

    The candidate arrives as a relative path from the tool call, so resolving it
    against the process CWD would miss the file entirely and make every rewrite
    look like a brand-new file. Falling back to the source tree is what lets the
    guard tell "rewrote the file, brake untouched" apart from "added a brake".
    """
    for candidate in (relative_path, SRC_DIR / relative_path, SRC_DIR / relative_path.name):
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return None


def _violations_against(
    name: str, new_text: str, old_text: Optional[str],
    whole_file: Optional[bool] = None,
) -> List[str]:
    """Returns the protected constructs ``new_text`` would add or alter.

    ``old_text`` is the trusted prior content, or None for a file that did not
    exist. Taking it as an argument rather than reading it is what lets the same
    comparison run against a candidate tool call *and* against what is actually
    sitting on disk, where the only prior version available is the baseline.

    ``whole_file`` says whether ``new_text`` is the complete file or just a
    fragment of it. It is inferred from the line counts when not given, which is
    the right guess for a candidate write but the wrong one for a file read back
    from disk: a file that was gutted to a fraction of its former length would
    then be mistaken for a small edit, and the brake it no longer defines would
    not be reported as removed.
    """
    if ALLOW_PROTECTED:
        return []
    found: List[str] = []
    if name in PROTECTED_FILES:
        found.append("%s (protected file)" % name)
    # Removal is only meaningful for a whole-file rewrite. A short fragment is
    # not a document that had a brake deleted from it; treating it as one would
    # make every small, legitimate write look like an attack on all four brakes.
    if whole_file is None:
        whole_file = old_text is not None and len(new_text.splitlines()) >= (
            0.5 * len(old_text.splitlines())
        )
    for filename, pattern in PROTECTED_SYMBOLS:
        if name != filename:
            continue
        new_blocks = _protected_blocks(new_text, pattern)
        if old_text is None:
            # New file: anything protected it defines is new.
            if new_blocks:
                found.append("%s::%s" % (filename, pattern))
            continue
        old_blocks = _protected_blocks(old_text, pattern)
        if new_blocks == old_blocks:
            # Bodies are byte-identical, so this is an ordinary edit elsewhere in
            # the file. Comparing the full tuple also catches a second definition
            # appended to shadow the protected one.
            continue
        if not new_blocks:
            # Only a whole-file rewrite can have removed a brake. A fragment
            # simply does not mention it, which is not the same thing.
            if old_blocks and whole_file:
                found.append("%s::%s (removed)" % (filename, pattern))
            continue
        found.append("%s::%s" % (filename, pattern))
    return found


def _violations(path: Path, new_text: str) -> List[str]:
    """Pre-dispatch check for a candidate write."""
    return _violations_against(path.name, new_text, _read_existing(path))


def protection_violations() -> Dict[str, List[str]]:
    """Protected constructs altered on disk relative to the baseline.

    ``guard_tool_call`` only sees ``file_write``, but the source tree is also
    reachable through ``bash_exec`` — a ``sed -i`` or a small Python script never
    goes near the guard. A brake weakened that way would still pass verification
    if the test suite stayed green, and would then be adopted as the new baseline.
    This re-runs the same comparison against the baseline, which is the last
    trusted copy, so the bypass is caught before the edit is blessed.
    """
    if ALLOW_PROTECTED:
        return {}
    base = _baseline_files()
    current = snapshot()
    found: Dict[str, List[str]] = {}
    for name, path in base.items():
        new_text = current.get(name)
        old_text = path.read_text(encoding="utf-8", errors="replace")
        if new_text is not None and new_text == old_text:
            # Byte-identical to the trusted baseline, so this file cannot have
            # altered a brake. This skip is load-bearing: the protected files are
            # themselves part of every baseline, so comparing them unconditionally
            # would report them as violations on every single self-patch and
            # refuse all of them.
            continue
        if new_text is None:
            # A protected file that was deleted outright.
            violations = _violations_against(name, "", old_text, whole_file=True)
        else:
            violations = _violations_against(name, new_text, old_text, whole_file=True)
        if violations:
            found[name] = violations
    for name, text in current.items():
        if name in base:
            continue
        violations = _violations_against(name, text, None)
        if violations:
            found[name] = violations
    return found


def is_protected_change(relative_path: str, new_text: str) -> List[str]:
    """Checks a candidate edit against the protected-construct list.

    Exposed separately from the runner so the host (or a test) can check an edit
    before it is written, not only after.
    """
    return _violations(Path(relative_path), new_text)


def within_src_tree(path: str) -> bool:
    """True when a tool call targeted JINX's own framework source."""
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return False
    try:
        return resolved.is_relative_to(SRC_DIR.resolve())
    except (OSError, ValueError, AttributeError):
        return str(resolved).startswith(str(SRC_DIR.resolve()))


def _target_for(name: str) -> Path:
    """Resolves a framework-relative baseline key back to its source file."""
    return SRC_DIR / name


def _tracked_files() -> Dict[str, Path]:
    """Framework Python files covered by the self-patch baseline."""
    tracked: Dict[str, Path] = {}
    if SRC_DIR.exists():
        for path in sorted(SRC_DIR.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tracked[path.relative_to(SRC_DIR).as_posix()] = path
    return tracked


def snapshot() -> Dict[str, str]:
    """Captures framework source for protection and rollback."""
    snap: Dict[str, str] = {}
    for name, path in _tracked_files().items():
        try:
            # Keys are paths relative to SRC_DIR, so same-named modules in
            # different subpackages cannot collide on restore.
            snap[name] = path.read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeDecodeError) as e:
            logger.warning("Could not snapshot %s: %s", path, e)
    return snap


def changed_files(snap: Dict[str, str]) -> List[str]:
    """Names of framework files that differ from the snapshot (or are new)."""
    changed: List[str] = []
    current = snapshot()
    for name, text in current.items():
        if name not in snap or snap[name] != text:
            changed.append(name)
    # A file deleted by the model is also a change, and a dangerous one.
    for name in snap:
        if name not in current:
            changed.append(name)
    return changed


def restore(snap: Dict[str, str], only: Optional[str] = None) -> List[str]:
    """Reverts tracked framework files to the snapshot. Returns files restored."""
    restored: List[str] = []
    current = snapshot()
    for name, text in snap.items():
        if only is not None and not name.startswith(only):
            continue
        if current.get(name) != text:
            target = _target_for(name)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                restored.append(name)
            except OSError as e:
                logger.error("Could not restore %s: %s", name, e)
    for name in current:
        if name in snap or (only is not None and not name.startswith(only)):
            continue
        try:
            _target_for(name).unlink(missing_ok=True)
            restored.append("%s (deleted)" % name)
        except OSError as e:
            logger.error("Could not remove new file %s: %s", name, e)
    if restored:
        logger.warning("Self-patch gate reverted: %s", ", ".join(restored))
    return restored


# --- on-disk baseline -------------------------------------------------------
# The runner is a one-transition-per-process state machine, so the baseline has
# to survive across processes. It deliberately does NOT live in the run state:
# that file is YAML persisted every round, and a snapshot of the whole source
# tree would make it grow with the codebase — the same unbounded-growth defect
# this feature is meant to coexist with.

BASELINE_DIR: Path = Path(
    os.environ.get(
        "JINX_SELF_PATCH_BASELINE",
        str(Path(__file__).resolve().parent.parent.parent / ".selfpatch_baseline"),
    )
)


_NO_LOCK_WARNED: set = set()


def _warn_no_lock(path: Path, why: str) -> None:
    """Say once per lock file (not once per round) that it cannot be used and work goes on without it."""
    key = str(path)
    if key not in _NO_LOCK_WARNED:
        _NO_LOCK_WARNED.add(key)
        logger.warning("%s: %s; continuing without the lock", key, why)


@contextlib.contextmanager
def file_lock(path: Path, timeout: float = 10.0) -> Iterator[bool]:
    """An exclusive lock between processes that disappears with its holder.

    Several JINX sessions can run at once and each transition is its own process, so two of
    them can reach the same read-modify-write at the same moment (the lesson ledger, the
    self-patch baseline). The lock is an OS file lock (``flock``, ``msvcrt.locking`` on
    Windows), so a process that is killed while holding it frees it: there is no stale lock to
    detect or break, which is what makes lock files fragile everywhere else.

    Yields True when the caller may go ahead and False when another process kept the lock for the
    whole ``timeout`` seconds; the caller then skips its cycle. A lock that cannot exist here at all
    (the lock file cannot be opened, or the file system or platform has no file locks) is not a lock
    that somebody else holds: it is warned about once and the caller goes ahead without it, as a
    single process would. Treating it as busy would switch the self-patch gate and the lesson ledger
    off for good. It never raises for a locking problem.
    """
    held = False
    handle = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+b")
    except OSError as e:
        _warn_no_lock(path, "cannot open the lock file: %s" % e)
        yield True
        return
    usable = True
    try:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError as e:
                busy = e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK)
                if not busy or time.monotonic() >= deadline:
                    if not busy:
                        usable = False
                        _warn_no_lock(path, "file locks are not usable here: %s" % e)
                    break
                time.sleep(0.02)
            except ImportError:
                usable = False
                _warn_no_lock(path, "this platform has no file locks")
                break
        yield held or not usable
    finally:
        if held:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, ImportError):
                pass
        handle.close()


def gate_lock(timeout: float = 30.0) -> "contextlib.AbstractContextManager[bool]":
    """The lock every baseline operation runs under (capture, verify-and-restore, clear).

    Two processes working on the baseline at once could leave it half written, and
    :func:`restore` deletes tracked files that the baseline does not list, so a half written
    baseline is not a harmless state.
    """
    return file_lock(BASELINE_DIR.parent / ".selfpatch.lock", timeout)


def _baseline_files() -> Dict[str, Path]:
    """Baseline files keyed by their path relative to the baseline directory.

    Relative keys, for the same reason ``snapshot`` uses them: a basename key
    collides across subpackages and would compare the wrong file against the
    wrong baseline.
    """
    if not BASELINE_DIR.exists():
        return {}
    return {
        p.relative_to(BASELINE_DIR).as_posix(): p
        for p in sorted(BASELINE_DIR.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def capture_baseline() -> bool:
    """Copies the tracked files to the baseline directory.

    Returns True when a usable baseline was written. A failure is logged and
    reported rather than raised: no baseline means no self-patching this round,
    which is the safe outcome.
    """
    if not SRC_DIR.exists():
        return False
    snap = snapshot()
    if not snap:
        return False
    try:
        if BASELINE_DIR.exists():
            shutil.rmtree(BASELINE_DIR, ignore_errors=True)
        # Written file by file rather than copied as a tree, because the baseline
        # spans two roots: the source package and the repository's tests.
        for name, text in snap.items():
            target = BASELINE_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return True
    except OSError as e:
        logger.error("Could not capture self-patch baseline: %s", e)
        return False


def baseline_changed() -> List[str]:
    """Framework files that differ from the on-disk baseline, or are new."""
    base = _baseline_files()
    if not base:
        return []
    current = snapshot()
    changed = [n for n, t in current.items() if n not in base or base[n].read_text(
        encoding="utf-8"
    ) != t]
    changed.extend(n for n in base if n not in current)
    return changed


def restore_baseline(only: Optional[str] = None) -> List[str]:
    """Restores the tracked files from the on-disk baseline.

    The baseline is deliberately KEPT afterwards. A previous version cleared it
    here, which disarmed the gate for the rest of the run: ``baseline_changed``
    returns nothing without a baseline, and a new one is only captured when a new
    session starts, so the second bad self-patch in the same run would go
    unchecked. A successful restore leaves the source equal to the baseline, so
    keeping it cannot trigger a spurious re-revert, and if the restore itself
    only partly succeeded the baseline is the only remaining copy of the working
    source -- dropping it there would make the damage permanent.
    """
    snap = {
        name: path.read_text(encoding="utf-8")
        for name, path in _baseline_files().items()
    }
    if not snap:
        return []
    return restore(snap, only=only)


def clear_baseline() -> None:
    """Drops the baseline directory once it is no longer needed."""
    shutil.rmtree(BASELINE_DIR, ignore_errors=True)


def _run(cmd: List[str], cwd: Path, timeout: int) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timed out after %ss" % timeout
    except OSError as e:
        return False, "could not launch: %s" % e
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


# Run in a child process: a module that fails (or hangs) on import must not take the verifier down
# with it, and the child sees the files as they are on disk, not the modules this process loaded
# long ago. It imports exactly what the next transition imports, every module of the package.
_IMPORT_CHECK = """\
import importlib, sys
from pathlib import Path
pkg = Path(sys.argv[1])
sys.path.insert(0, str(pkg.parent))
failed = []
for stem in ["__init__"] + sorted(p.stem for p in pkg.glob("*.py") if p.stem != "__init__"):
    name = pkg.name if stem == "__init__" else pkg.name + "." + stem
    try:
        importlib.import_module(name)
    except BaseException as e:
        failed.append("%s: %s: %s" % (name, type(e).__name__, e))
print("\\n".join(failed))
sys.exit(1 if failed else 0)
"""


def verify(timeout: int = VERIFY_TIMEOUT) -> Dict[str, Any]:
    """Runs the only automatic self-patch checks: the source compiles, and every module imports.

    Functional tests belong to the editor CLI and the task being performed. JINX
    must not repeat a project's suite after every framework edit, but it must not
    bless source that cannot be imported on the next transition either. Compiling
    alone accepts a module with a missing import or an exception at load time, and
    such an edit would then become the baseline every later rollback returns to.
    """
    # -f: without it compileall trusts a .pyc whose recorded modification time (whole seconds) matches
    # the source's, so an edit made within the same second as the version compiled before it goes unseen.
    ok, output = _run(
        [sys.executable, "-m", "compileall", "-q", "-f", str(SRC_DIR)], SRC_DIR, timeout
    )
    checks = [{
        "name": "compileall",
        "ok": ok,
        "tail": "\n".join(output.strip().splitlines()[-25:]),
    }]
    summary = "compileall %s" % ("PASS" if ok else "FAIL")
    if ok:
        # Only worth running on source that compiles: otherwise it would repeat the same error.
        # The working directory is the one above the package, so it is the only thing the
        # interpreter puts on the path besides the package's own dependencies.
        # A cache directory of its own makes the child compile what is on disk now, whatever the
        # project's own .pyc files say, and leaves those files alone.
        with tempfile.TemporaryDirectory(prefix="jinx-verify-") as cache:
            imported, output = _run(
                [sys.executable, "-X", "pycache_prefix=" + cache, "-c", _IMPORT_CHECK, str(SRC_DIR)],
                SRC_DIR.parent, timeout,
            )
        checks.append({
            "name": "import",
            "ok": imported,
            "tail": "\n".join(output.strip().splitlines()[-25:]),
        })
        summary += ", import %s" % ("PASS" if imported else "FAIL")
        ok = imported
    return {"ok": ok, "checks": checks, "summary": summary}


def guard_tool_call(path: str, content: str) -> Optional[str]:
    """Returns a rejection reason if a tool call would touch brake logic.

    Called before a write is dispatched, so a protected edit is refused rather
    than written-then-undone.
    """
    if not SELF_PATCH_ENABLED or ALLOW_PROTECTED:
        return None
    if not within_src_tree(path):
        return None
    violations = is_protected_change(path, content)
    if not violations:
        return None
    return prompts.PROTECTED_EDIT_REFUSAL % (Path(path).name, "; ".join(violations))
