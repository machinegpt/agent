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
"""Which editor is calling? A bare ``python .agent/jinx.py`` has to find the caller's own session.

Every call is a new process, so nothing in memory says who it is. What survives from one call to the
next is the *editor*: the CLI or window that starts JINX again and again. Its process, and the
terminal and shells above it, are the same on every call, while the shell it starts for one command
dies with the command. So a session remembers a **fingerprint** of its editor (the ancestors of the
JINX process and a few environment variables that name a terminal pane or a window), and a later call
is matched against those fingerprints.

The match is deliberately strict, because a wrong answer would hand one agent another agent's work:

* a session matches only when *all* of its distinguishing tokens are present in the caller's;
* tokens every unfinished session has (the terminal multiplexer, the shell, the host's own manager)
  distinguish nobody and are ignored;
* two sessions of one editor process (sub-agents of one window share it) cannot be told apart, so
  neither matches: the caller is asked for ``--session <id>`` instead of being guessed. The same goes for
  a session whose tokens another session's tokens contain (see :func:`identify`).

Nothing here raises: an unreadable /proc or a missing ``ps`` just means "no fingerprint", and the
explicit ``--session <id>`` always works.
"""

import os
import subprocess
import sys
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Environment variables that name the terminal pane, window or editor instance the process runs in.
# They are compared as ``NAME=value``: the same value is a hint, a different value is a different place.
HINT_ENV = (
    "CLAUDE_PID", "CLAUDE_CODE_SESSION_ID",
    "TMUX_PANE", "STY", "ZELLIJ_PANE_ID", "KITTY_WINDOW_ID", "WEZTERM_PANE",
    "ITERM_SESSION_ID", "TERM_SESSION_ID", "WT_SESSION", "SSH_TTY",
    "VSCODE_IPC_HOOK_CLI", "CURSOR_TRACE_ID",
)
MAX_ANCESTORS = 16         # a fingerprint is a short list: a deep process tree is cut above this
MAX_ENV = 8
ENV_VALUE_CAP = 60


def _read_proc(pid: int) -> Optional[Tuple[int, int]]:
    """(parent pid, start time) of a process, from /proc; None when it is gone or unreadable."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    end = raw.rfind(")")                       # the command name may hold spaces and parentheses
    fields = raw[end + 2:].split() if end > 0 else []
    try:
        return int(fields[1]), int(fields[19])
    except (IndexError, ValueError):
        return None


def _read_ps(pid: int) -> Optional[Tuple[int, int]]:
    """The same through ``ps``, for systems without /proc (macOS, BSD). The start time is a checksum of ps's text."""
    try:
        out = subprocess.run(["ps", "-o", "ppid=,lstart=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.split(None, 1)
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    return int(parts[0]), zlib.crc32(parts[1].encode("utf-8", "replace"))


def ancestors(pid: int) -> List[Tuple[int, int]]:
    """[(pid, start time)] of the ancestors of ``pid``, nearest first, stopping below init."""
    reader = _read_proc if os.path.isdir("/proc/self") else _read_ps
    out: List[Tuple[int, int]] = []
    seen = {pid}
    node = reader(pid)
    while node is not None and len(out) < MAX_ANCESTORS:
        ppid = node[0]
        if ppid <= 1 or ppid in seen:
            break
        seen.add(ppid)
        parent = reader(ppid)
        if parent is None:
            break
        out.append((ppid, parent[1]))
        node = parent
    return out


def fingerprint(pid: Optional[int] = None) -> List[str]:
    """What identifies the editor behind this process; empty when nothing can be told.

    The process itself is left out (it is new on every call). Its parent is kept: for an editor that
    starts JINX directly it *is* the editor, and for one that goes through a shell per command that
    token simply never matches again, which :func:`refine` drops after the second call.
    """
    try:
        chain = ancestors(os.getpid() if pid is None else pid)
    except Exception:                                       # a fingerprint is a hint, never a reason to fail
        chain = []
    toks = [f"p{apid}.{start}" for apid, start in chain]
    env = []
    for name in HINT_ENV:
        val = os.environ.get(name, "").strip()
        if val:
            env.append(f"e{name}={val[:ENV_VALUE_CAP]}")
    return toks + env[:MAX_ENV]


def refine(old: Sequence[str], now: Sequence[str], seen: int) -> Tuple[List[str], int]:
    """Keep what the editor showed on *both* calls: the tokens of a dead per-command shell fall away.

    Returns the new token list and how many calls it has survived. When nothing is shared, a different
    editor has taken the session over (a person resumed it from another terminal): start over.
    """
    if not now:
        return list(old), seen
    if not old:
        return list(now), 1
    now_set = set(now)
    shared = [t for t in old if t in now_set]
    if shared:
        return shared, seen + 1
    return list(now), 1


def identify(caller: Iterable[str], candidates: Dict[str, Sequence[str]]) -> Tuple[Optional[str], List[str]]:
    """The session the caller belongs to, among the unfinished ``candidates`` ({id: tokens}).

    Returns ``(id, matches)``: ``id`` is set only when exactly one session can be it. ``matches`` lists every
    session whose own tokens the caller carries, so that an ambiguous caller can be told which ones.

    A session whose own tokens another session's tokens contain is never the answer: whoever started the
    other one carries them all, so the two cannot be told apart (an editor that has just started a second
    session has not yet been seen twice, and its second session still holds the tokens of the first).
    """
    have = set(caller)
    sets = {sid: set(toks) for sid, toks in candidates.items() if toks}
    if not sets or not have:
        return None, []
    core: Set[str] = set.intersection(*sets.values()) if len(sets) > 1 else set()
    own = {sid: toks - core for sid, toks in sets.items()}
    matches = sorted(sid for sid, t in own.items() if t and t <= have)
    clear = [sid for sid in matches if not any(o != sid and own[sid] <= own[o] for o in own)]
    return (clear[0] if len(clear) == 1 else None), matches


if __name__ == "__main__":        # python identity.py: what this process would be recognised by
    print("\n".join(fingerprint()) or "(no fingerprint)", file=sys.stdout)
