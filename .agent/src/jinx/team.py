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
"""Looking at the other sessions: what each one does, what it changed, and the briefing for a helper.

Nothing here changes anything. It reads what ``mail.py`` and ``sessions.py`` keep for every session (the
presence card, the ``.mbox`` and ``.yaml`` of every round, and the file that remembers the round still in
progress) and prints a short text. An agent asks for it with one command,

    python .agent/jinx.py --peek <id> [--diff] [--full]

so the tokens are spent only when it wants to know. ``--peek`` alone lists every session in one line each.
The same data, cut down to a few lines, is what a session gets once when it joins another one to help
(``--join <id>``, see :func:`briefing`).

Everything is bounded: a peek without ``--full`` never prints more than MAX_OUT characters, and a diff
shows the newest changes first, so that what is cut is what is oldest.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import mail

MAX_OUT = 12_000                # characters printed without --full
MAX_FULL = 400_000              # ... and with it: a diff nobody can read is no help either
DIFF_LINES = 160                # lines of diff shown with --diff
FILE_DIFF_LINES = 80            # ... of which at most this many per file
DIFF_ROUNDS = 3                 # rounds whose diff --diff shows (newest first)
ROUNDS_SHOWN = 8                # rounds listed in the status
LIST_SHOWN = 12                 # sessions listed by a bare --peek
MAX_FILE_BYTES = 400_000        # a larger file is only counted, not diffed
MAX_PATCH_BYTES = 5_000_000     # a larger round .mbox is listed from its .yaml only (no diff, no checksum)


# --------------------------------------------------------------------------- reading what a session left
def session_dir(office: mail.Office, sid: str) -> Optional[Path]:
    """The directory of a session, also when it has been archived (finished more than a week ago)."""
    if not mail._SID_RE.match(sid):
        return None
    d = office.sessions_dir / sid
    if d.is_dir():
        return d
    try:
        hits = sorted(p for p in (office.root / "archive").glob(f"{sid}-*") if p.is_dir())
    except OSError:
        hits = []
    return hits[-1] if hits else None


def _load(path: Path, limit: int = 1_000_000) -> Dict[str, Any]:
    try:
        data = mail.load_yaml(path.read_text("utf-8"), limit=limit)
    except (OSError, mail.InvalidMail):
        return {}
    return data if isinstance(data, dict) else {}


def read_card(d: Path) -> Dict[str, Any]:
    return _load(d / "card.yaml")


def read_state(d: Path) -> Dict[str, Any]:
    """The state block the session's model keeps (``JINX.yaml``): its own words for the task, facts, open items."""
    data = _load(d / "JINX.yaml")
    st = data.get("state")
    return st if isinstance(st, dict) else {}


def _stat(diff: str) -> Dict[str, Tuple[int, int]]:
    """{path: (added, removed)} lines of a git diff."""
    out: Dict[str, List[int]] = {}
    cur: Optional[str] = None
    hunk = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            cur = m.group(2) if m else line[11:]
            out.setdefault(cur, [0, 0])
            hunk = False
        elif line.startswith("@@"):
            hunk = True
        elif hunk and cur is not None:
            if line.startswith("+"):
                out[cur][0] += 1
            elif line.startswith("-"):
                out[cur][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def _chunks(diff: str) -> List[Tuple[str, List[str]]]:
    """A git diff split per file: [(path, its lines)]."""
    out: List[Tuple[str, List[str]]] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            out.append((m.group(2) if m else line[11:], [line]))
        elif out:
            out[-1][1].append(line)
    return out


@dataclass
class RoundInfo:
    n: int
    summary: str
    files: List[str]
    plus: int = 0
    minus: int = 0
    diff: str = ""
    intact: bool = True
    flags: List[str] = field(default_factory=list)       # what the round's record says about its own limits


def rounds_of(d: Path) -> List[RoundInfo]:
    """The rounds of a session that changed something, oldest first (a round with no change leaves no .mbox)."""
    out: List[RoundInfo] = []
    try:
        names = sorted(p for p in (d / "rounds").iterdir() if p.name.endswith(".mbox"))
    except OSError:
        return out
    for p in names:
        record = mail.read_record(p.with_suffix(".yaml"))
        try:
            n = int(record["round"] if record.get("round") is not None else p.name.split(".")[0])
        except (KeyError, TypeError, ValueError):
            n = 0
        big = False
        try:
            big = p.stat().st_size > MAX_PATCH_BYTES
            data = b"" if big else p.read_bytes()
        except OSError:
            continue
        fields, body = mail.split_mbox(data)
        text = body.decode("utf-8", "replace")
        stat = _stat(text)
        want = record.get("sha256")
        files = [str(f) for f in (record.get("touches") or [])] or sorted(stat)
        summary = record.get("summary") or fields.get("subject", "").removeprefix("[PATCH] ")
        out.append(RoundInfo(
            n=n, summary=mail._short(summary, mail.MAX_SUMMARY), files=files,
            plus=sum(a for a, _ in stat.values()), minus=sum(r for _, r in stat.values()), diff=text,
            intact=big or not want or want == mail.checksum(data), flags=[f for f in mail.FLAGS if record.get(f)]))
    return out


def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def _udiff(rel: str, before: Optional[bytes], after: Optional[bytes]) -> str:
    """A unified diff of two versions of a file; '' when they are the same."""
    if before == after:
        return ""
    if (before is not None and _is_binary(before)) or (after is not None and _is_binary(after)):
        return f"diff --git a/{rel} b/{rel}\nBinary file changed\n"
    a = (before or b"").decode("utf-8", "replace").splitlines(keepends=True)
    b = (after or b"").decode("utf-8", "replace").splitlines(keepends=True)
    body = "".join(l if l.endswith("\n") else l + "\n\\ No newline at end of file\n"
                   for l in difflib.unified_diff(a, b, f"a/{rel}" if before is not None else "/dev/null",
                                                 f"b/{rel}" if after is not None else "/dev/null", n=3))
    return f"diff --git a/{rel} b/{rel}\n{body}" if body else ""


def in_progress(root: Path, d: Path) -> List[Tuple[str, str]]:
    """What the session has changed in the round it is still in, as [(path, diff)]; not in a round file yet.

    The round remembers the state of every file at the moment the session took it. Comparing that with the
    file now is the whole answer: nothing is written, and nothing about it needs git except reading the
    old content out of the object database.
    """
    base = _load(d / "round.yaml").get("base")
    out: List[Tuple[str, str]] = []
    if not isinstance(base, dict):
        return out
    for rel, ent in sorted(base.items()):
        try:
            rel = mail.check_relpath(rel)
        except mail.InvalidMail:
            continue
        before: Optional[bytes] = None
        if ent:
            try:
                before = mail.blob_bytes(root, str(ent[1]), MAX_FILE_BYTES)
            except (mail.MailError, OSError, IndexError):
                before = None
            if before is None:
                out.append((rel, f"{rel}: changed, but the old version is not available here\n"))
                continue
        full = root / rel
        try:
            after = full.read_bytes() if full.is_file() and not full.is_symlink() and full.stat().st_size <= MAX_FILE_BYTES else None
        except OSError:
            after = None
        if ent is not None and before is not None and not full.exists():
            after = None
        text = _udiff(rel, before, after)
        if text:
            out.append((rel, text))
    return out


# --------------------------------------------------------------------------- one line per session, and the status
def _ago(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{round(s / 60)}m"
    if s < 172800:
        return f"{round(s / 3600)}h"
    return f"{round(s / 86400)}d"


def state_word(card: Dict[str, Any], now: float) -> str:
    """'working', 'finished', or 'silent 20m' for a session whose editor stopped calling."""
    if not card:
        return "unknown"
    if card.get("state") == "done":
        return "finished"
    age = now - float(card.get("hb") or 0)
    if age <= float(card.get("lease_s") or mail.DEFAULT_LEASE_S):
        return "working"
    return f"silent {_ago(age)}"


def _line(sid: str, card: Dict[str, Any], now: float, mine: Sequence[str] = ()) -> str:
    word = state_word(card, now)
    head = f"{sid} {word}"
    if card.get("state") != "done" and word == "working":
        head += f" {_ago(now - float(card.get('hb') or now))} ago"
    return f"{head}; " + mail.peer_line({**card, "id": sid}, mine)


def board(office: mail.Office, *, now: Optional[float] = None, cmd: Optional[str] = None) -> str:
    """Every session in one line: who runs, what each does, what it holds. The newest finished ones follow."""
    now = time.time() if now is None else now
    cards = {sid: read_card(office.sessions_dir / sid) for sid in office.session_ids()}
    live = sorted((s for s, c in cards.items() if c and c.get("state") != "done"), key=lambda s: (s,))
    done = sorted((s for s, c in cards.items() if c and c.get("state") == "done"),
                  key=lambda s: -float(cards[s].get("hb") or 0))
    if not live and not done:
        return "no JINX sessions yet"
    lines = [f"{len(live)} running" + (f", {len(done)} finished" if done else "")]
    for sid in (live + done)[:LIST_SHOWN]:
        lines.append(_line(sid, cards[sid], now))
    if len(live) + len(done) > LIST_SHOWN:
        lines.append(f"(+{len(live) + len(done) - LIST_SHOWN} older)")
    lines.append(f"details and changes of one: {cmd or office.cmd} --peek <id> [--diff]")
    return "\n".join(lines)


def _emit(lines: List[str], text_lines: List[str], budget: List[Optional[int]], per_file: Optional[int],
          cut: List[bool]) -> bool:
    """Append diff lines within the budget (``cut[0]`` records that something was left out); False once it is spent."""
    left = budget[0]
    cap = len(text_lines) if per_file is None else min(len(text_lines), per_file)
    if left is not None:
        cap = min(cap, left)
    lines.extend(text_lines[:cap])
    if cap < len(text_lines):
        lines.append(f"... {len(text_lines) - cap} more lines")
        cut[0] = True
    if left is not None:
        budget[0] = left - cap
        return budget[0] > 0
    return True


def status(office: mail.Office, root: Path, sid: str, *, diff: bool = False, full: bool = False,
           now: Optional[float] = None, cmd: Optional[str] = None) -> str:
    """What one session does and has done; with ``diff`` also what it changed, newest first.

    Raises KeyError for a session that does not exist.
    """
    d = session_dir(office, sid)
    if d is None:
        known = ", ".join(office.session_ids()) or "none"
        raise KeyError(f"no session {sid!r} (known: {known})")
    now = time.time() if now is None else now
    card, st = read_card(d), read_state(d)
    cmd = cmd or office.cmd
    word = state_word(card, now)
    head = f"{sid} {word}, round {int(card.get('round') or 0)}"
    if card.get("editor"):
        head += f" ({mail._short(card['editor'], 16)})"
    if card.get("room") and card["room"] != sid:
        head += f", team of {card['room']}"
    lines = [head, "task: " + mail._short(st.get("task") or card.get("task"), 200)]
    if card.get("doing"):
        lines.append("doing: " + mail._short(card["doing"], 120))
    holds = sorted((card.get("holds") or {}).keys())
    if holds:
        lines.append("holds: " + mail._names(holds, 6))
    rounds = rounds_of(d)
    if rounds:
        shown = rounds if full else rounds[-ROUNDS_SHOWN:]
        lines.append(f"changed in {len(rounds)} round(s)" + (f", last {len(shown)}:" if len(shown) < len(rounds) else ":"))
        for r in shown:
            tail = f" [{', '.join(r.flags)}]" if r.flags else ""
            tail += " [.mbox was edited]" if not r.intact else ""
            lines.append(f"  r{r.n} +{r.plus} -{r.minus} {mail._names(r.files, 4)}"
                         + (f": {mail._short(r.summary, 70)}" if r.summary else "") + tail)
    else:
        lines.append("no round has changed a file yet")
    pending = in_progress(root, d) if word == "working" or card.get("holds") else []
    if pending:
        lines.append("not yet in a round file (the round is still running):")
        for rel, text in pending:
            a, r = _stat(text).get(rel, (0, 0))
            lines.append(f"  {rel} +{a} -{r}")
    if st.get("open"):
        lines.append("open: " + " | ".join(mail._short(x, 110) for x in list(st["open"])[:2]))
    if diff or full:
        budget: List[Optional[int]] = [None if full else DIFF_LINES]
        cut = [False]
        out: List[str] = []
        ok = True
        for rel, text in pending:
            if not ok:
                break
            out.append(f"=== {rel} (not yet in a round file)")
            ok = _emit(out, text.splitlines(), budget, None if full else FILE_DIFF_LINES, cut)
        shown = rounds if full else rounds[-DIFF_ROUNDS:]
        for r in reversed(shown):
            if not ok:
                break
            out.append(f"=== round {r.n}" + (f": {mail._short(r.summary, 70)}" if r.summary else ""))
            for rel, chunk in _chunks(r.diff):
                if not ok:
                    break
                ok = _emit(out, chunk, budget, None if full else FILE_DIFF_LINES, cut)
        if not ok or cut[0] or len(shown) < len(rounds):
            out.append(f"(not everything is shown; all of it: {cmd} --peek {sid} --full)")
        lines.extend(out)
    else:
        lines.append(f"the changes themselves: {cmd} --peek {sid} --diff")
    text = "\n".join(lines)
    limit = MAX_FULL if full else MAX_OUT
    if len(text) > limit:
        text = text[:limit].rstrip() + f"\n[cut at {limit} characters" + ("" if full else f"; {cmd} --peek {sid} --full for all") + "]"
    return text


# --------------------------------------------------------------------------- the briefing of a helper
def briefing(office: mail.Office, root: Path, host: str, cmd: Optional[str] = None) -> Dict[str, Any]:
    """What a session that joins ``host`` is told once, before its first step, as a few short lines.

    The task in the host's own words, what its rounds changed, what it holds now and its open questions:
    enough to start helping without reading anything, and where to look for the rest.
    """
    d = session_dir(office, host)
    card = read_card(d) if d else {}
    st = read_state(d) if d else {}
    cmd = cmd or office.cmd
    out: Dict[str, Any] = {"host": host, "task": mail._short(st.get("task") or card.get("task"), 160)}
    if card.get("doing"):
        out["doing"] = mail._short(card["doing"], 100)
    holds = sorted((card.get("holds") or {}).keys())
    if holds:
        out["holds"] = mail._names(holds, 4)
    done = [f"r{r.n} {mail._names(r.files, 2)}: {mail._short(r.summary, 60)}".rstrip(": ")
            for r in (rounds_of(d)[-3:] if d else [])]
    if done:
        out["done"] = done
    opened = [mail._short(x, 110) for x in list(st.get("open") or [])[:2]]
    if opened:
        out["open"] = opened
    out["more"] = f"{cmd} --peek {host} --diff"
    out["mail"] = "to: room reaches the host (it was told you joined) and everybody else in the team"
    return out


# --------------------------------------------------------------------------- command line
def peek_main(argv: Sequence[str], agent_dir: Path) -> int:
    """``jinx.py --peek [id] [--diff] [--full]``: print and return an exit code. Changes nothing."""
    ap = argparse.ArgumentParser(prog="jinx.py", add_help=False)
    ap.add_argument("--peek", nargs="?", const="*", default=None)
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--full", action="store_true")
    args, _ = ap.parse_known_args(list(argv))
    ipc = Path(os.environ.get("JINX_IPC_DIR") or (Path(agent_dir) / "ipc"))
    root = Path(os.environ.get("JINX_PROJECT_ROOT") or Path(agent_dir).parent).resolve()
    if not (ipc / "sessions").is_dir():
        print("no JINX sessions yet")
        return 0
    try:
        label = Path(os.path.relpath(ipc, root)).as_posix()
        rel_agent = Path(os.path.relpath(agent_dir, root)).as_posix()
        office = mail.Office(ipc, label=label, cmd=f"python {rel_agent}/jinx.py")
    except (OSError, ValueError, mail.MailError) as e:
        print(f"cannot read {ipc}: {e}")
        return 1
    try:
        if args.peek in (None, "*", ""):
            print(board(office))
        else:
            print(status(office, root, args.peek.strip(), diff=args.diff, full=args.full))
    except KeyError as e:
        print(str(e.args[0]) if e.args else "no such session")
        return 1
    except (OSError, mail.MailError) as e:
        print(f"could not read the session: {e}")
        return 1
    return 0
