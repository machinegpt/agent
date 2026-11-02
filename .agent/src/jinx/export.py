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
"""The rounds of JINX sessions as one .mbox and one .yaml: what ``jinx.py --export`` writes.

Every round that changed files leaves ``rounds/NNNN.mbox`` and ``rounds/NNNN.yaml`` in its session's folder
(``mail.py``, "round files"). Collected in the order the rounds closed they make

    .agent/export/jinx-<session>.mbox    the series: `git am` applies it, one commit per round
    .agent/export/jinx-<session>.yaml    its record: the rounds, the checksums, and whether the series applies

    python .agent/jinx.py --export [SESSION ...] [--out DIR]

With no session every session of the project that is not archived is collected (``jinx-all``); an archived
one can be named. A session whose run ended on its own terms is exported by itself, without being asked
(``sessions.end``), and the runner prints one ``[JINX_EXPORT]`` line for the editor to pass on.

Nothing in the project changes: whether the series applies to HEAD is checked on a private copy of the index,
the way ``git am --3way`` would apply it, with and without the CR line endings a diff may carry (see
``check_series``). A round that cannot be passed on is left out and named, never silently: one whose .mbox no
longer matches the checksum its .yaml holds, one that could not be recorded, and one written by an older JINX.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import mail, team

logger = logging.getLogger("jinx.export")

EXPORT_DIR = "export"           # under .agent/
MAX_DETAIL = 240                # characters of git's complaint kept in the record
MAX_LABEL_SESSIONS = 3          # more sessions than this are called "all" in the file name
_DROP = ("jinx", "kind", "patch", "size")     # what a round's record says that the series does not need to repeat


@dataclass
class Found:
    """One round file pair as found on disk."""
    sid: str
    n: int
    closed: float
    data: bytes                                  # the .mbox
    record: Dict[str, Any]                       # the .yaml ({} when it is missing)
    verified: bool                               # False: no checksum to compare with


@dataclass
class Result:
    sessions: List[str]
    rounds: List[Found] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    mbox: Optional[Path] = None
    yaml: Optional[Path] = None
    applies: str = "unknown"                     # yes | already | together | no | unknown
    detail: str = ""                             # why not, when it is not "yes"
    keep_cr: bool = False                        # `git am` needs --keep-cr: the diffs carry CR line endings that count
    head: Optional[str] = None                   # HEAD when the first round began: the commit it applies to
    all_cmd: str = ""                            # the command that collects every session (for "together")

    @property
    def clean(self) -> bool:
        return bool(self.rounds) and not self.skipped and self.applies in ("yes", "already", "together")


# --------------------------------------------------------------------------- finding the rounds
def _number(path: Path, record: Dict[str, Any]) -> int:
    try:
        return int(record["round"] if record.get("round") is not None else path.name.split(".")[0])
    except (KeyError, TypeError, ValueError):
        return 0


def _stem_number(stem: str, record: Dict[str, Any]) -> int:
    try:
        return int(record["round"] if record.get("round") is not None else stem)
    except (KeyError, TypeError, ValueError):
        return 0


def find_rounds(office: mail.Office, sids: Sequence[str]) -> Tuple[List[Found], List[Dict[str, Any]]]:
    """(rounds in the order they closed, rounds that were left out and why) of these sessions."""
    found: List[Found] = []
    skipped: List[Dict[str, Any]] = []
    for sid in sids:
        d = team.session_dir(office, sid)
        if d is None:
            continue
        try:
            names = sorted(p.name for p in (d / "rounds").iterdir() if p.is_file())
        except OSError:
            continue
        stems: Dict[str, List[str]] = {}
        for name in names:
            stem, dot, ext = name.partition(".")
            if dot and ext in ("mbox", "yaml", "patch"):
                stems.setdefault(stem, []).append(ext)
        for stem, kinds in stems.items():
            p = d / "rounds" / f"{stem}.mbox"
            record = mail.read_record(p.with_suffix(".yaml")) if "yaml" in kinds else {}
            n = _stem_number(stem, record)
            if "mbox" not in kinds:
                # Nothing to export, but the work is not to vanish unnoticed from the series.
                if "patch" in kinds:
                    why = "a round file of an older JINX (.patch): it is not part of the export"
                elif record.get("failed"):
                    why = f"it changed files but could not be recorded: {record['failed']}"
                else:
                    why = "its .mbox is missing"
                skipped.append({"session": sid, "round": n, "why": why})
                continue
            try:
                data = p.read_bytes()
                closed = float(record.get("closed") or p.stat().st_mtime)
            except (OSError, TypeError, ValueError) as e:
                skipped.append({"session": sid, "round": n, "why": f"cannot be read: {e}"})
                continue
            want = record.get("sha256")
            if want and want != mail.checksum(data):
                skipped.append({"session": sid, "round": n, "why": "the .mbox no longer matches the checksum in "
                                                                    "its .yaml: it was changed after the round wrote it"})
                continue
            found.append(Found(sid, n, closed, data, record, bool(want)))
    found.sort(key=lambda f: (f.closed, f.sid, f.n))
    skipped.sort(key=lambda x: (x["session"], x["round"]))
    return found, skipped


# --------------------------------------------------------------------------- does the series apply?
def _git_ok(repo: Path, *args: str, env: Optional[Dict[str, str]] = None) -> bool:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=env)
    return r.returncode == 0


def _apply_chain(repo: Path, diffs: List[bytes], reverse: bool,
                 three_way: bool = False) -> Tuple[Optional[Tuple[int, str]], List[int]]:
    """(failure, no-ops): failure is None when every diff applies in turn to a private copy of HEAD's index, else
    (position, git's complaint); no-ops are the positions of diffs that changed nothing (only looked for with
    ``three_way``).

    ``three_way`` lets a diff whose context has moved still apply, by a merge on the blobs it names (the
    fallback of ``git am --3way``); what git cannot merge cleanly counts as not applying. A change HEAD holds
    already merges "cleanly" into nothing, which is why the no-ops are reported: ``git am`` stops at such a patch.
    """
    noops: List[int] = []
    with tempfile.TemporaryDirectory(prefix="jinx-export-") as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        has_head = _git_ok(repo, "rev-parse", "--verify", "-q", "HEAD")
        r = subprocess.run(["git", "-C", str(repo), "read-tree", "HEAD" if has_head else "--empty"],
                           capture_output=True, env=env)
        if r.returncode != 0:
            raise mail.MailError("git read-tree failed: " + r.stderr.decode(errors="replace").strip()[:MAX_DETAIL])

        def tree() -> str:
            w = subprocess.run(["git", "-C", str(repo), "write-tree"], capture_output=True, env=env)
            return w.stdout.decode().strip() if w.returncode == 0 else ""

        order = list(enumerate(diffs))
        if reverse:
            order.reverse()
        before = tree() if three_way else ""
        for i, diff in order:
            cmd = ["git", "-C", str(repo), "-c", "core.autocrlf=false", "apply", "--cached"]
            cmd += (["-R"] if reverse else []) + (["--3way"] if three_way else [])
            r = subprocess.run(cmd, input=diff, capture_output=True, env=env)
            if r.returncode != 0:
                return (i, " ".join(r.stderr.decode(errors="replace").split())[:MAX_DETAIL]), noops
            if three_way:
                after = tree()
                if after == before:
                    noops.append(i)
                before = after
    return None, noops


def _without_cr(diff: bytes) -> bytes:
    """The diff as ``git am`` reads it unless told ``--keep-cr``: a CR before a line feed is dropped."""
    return diff.replace(b"\r\n", b"\n")


def check_series(repo: Path, rounds: Sequence[Found]) -> Tuple[str, str, bool]:
    """(verdict, detail, keep_cr): whether ``git am --3way`` would apply these rounds to HEAD, and with ``--keep-cr``.

    The verdict is "yes", "already" (HEAD holds these changes), "no" (with the first round that fails) or "unknown".
    Nothing is changed: the diffs go through ``git apply --cached`` on a private copy of HEAD's index.

    Sessions share one working tree, so a round of one can build on a file another session changed first: its
    own rounds alone may not apply to HEAD, while all the sessions' rounds together do (see ``build``).

    A diff is made from the bytes of the files, so it carries CR line endings where a file has them. ``git am``
    drops them unless told ``--keep-cr``, which is wanted when the repository holds CRLF files as they are, and
    wrong when it holds LF and a checkout converts (``core.autocrlf``). Both readings are tried; ``keep_cr`` is
    the one that fits HEAD, and the plain one when both do. "Already" is asked before the merge is tried,
    because a merge of a change HEAD holds already "succeeds" and changes nothing.
    """
    diffs = [mail.split_mbox(f.data)[1] for f in rounds]
    plain = [_without_cr(d) for d in diffs]
    has_cr = plain != diffs
    variants = [(True, diffs), (False, plain)] if has_cr else [(False, diffs)]
    try:
        for keep, ds in variants:
            if _apply_chain(repo, ds, reverse=False)[0] is None:
                return "yes", "", keep
        for keep, ds in variants:
            if _apply_chain(repo, ds, reverse=True)[0] is None:
                return "already", "HEAD already holds these changes", keep
        failures = []
        for keep, ds in variants:
            bad, noops = _apply_chain(repo, ds, reverse=False, three_way=True)
            if bad is not None:
                failures.append(bad)
            elif len(noops) == len(ds):
                return "already", "HEAD already holds these changes", keep
            elif noops:
                f = rounds[noops[0]]
                return "no", (f"round {f.n} of {f.sid} is in HEAD already, and `git am` stops at a patch that changes "
                              "nothing (`git am --skip` goes on)"), keep
            else:
                return "yes", "", keep
        i, why = max(failures, key=lambda b: b[0])      # the reading that got further is the one that was meant
        return "no", f"round {rounds[i].n} of {rounds[i].sid} does not apply to HEAD: {why}", has_cr
    except (OSError, mail.MailError) as e:
        return "unknown", f"could not check: {e}", has_cr


# --------------------------------------------------------------------------- writing
def label_of(sids: Sequence[str]) -> str:
    return "-".join(sids) if 0 < len(sids) <= MAX_LABEL_SESSIONS else "all"


def _entry(f: Found) -> Dict[str, Any]:
    rest = {k: v for k, v in f.record.items() if k not in _DROP and k not in ("session", "round")}
    out: Dict[str, Any] = {"session": f.sid, "round": f.n, **rest}
    if not f.verified:
        out["unverified"] = "no .yaml record, so no checksum to compare with"
    return out


def build(office: mail.Office, root: Path, sids: Sequence[str], out_dir: Path, *, check: bool = True) -> Result:
    """Collect the rounds of ``sids`` into ``out_dir``/jinx-<label>.mbox and .yaml. Nothing is written when there are none."""
    found, skipped = find_rounds(office, sids)
    res = Result(sessions=sorted({f.sid for f in found}), rounds=found, skipped=skipped)
    if not found:
        return res
    res.head = next((str(f.record["head"]) for f in found if f.record.get("head")), None)
    res.keep_cr = any(b"\r" in mail.split_mbox(f.data)[1] for f in found)      # what it is when nothing is checked
    res.all_cmd = f"{office.cmd} --export"
    if check:
        res.applies, res.detail, res.keep_cr = check_series(root, found)
        if res.applies == "no":
            res.applies, res.detail = _together(office, root, found, sids, res)
    # A blank line before every "From ": that is how an mbox is told apart into messages.
    series = b"\n".join(f.data for f in found)
    name = "jinx-" + label_of(res.sessions)
    res.mbox, res.yaml = out_dir / (name + ".mbox"), out_dir / (name + ".yaml")
    out_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {
        "jinx": mail.PROTOCOL, "kind": "export", "made": mail.utc_iso(), "sessions": res.sessions,
        "mbox": res.mbox.name, "sha256": mail.checksum(series), "size": len(series), "count": len(found),
        "applies": res.applies}
    if res.detail:
        record["detail"] = res.detail
    if res.head:
        record["head"] = res.head
    if res.skipped:
        record["skipped"] = res.skipped
    record["apply"] = apply_command(res, root)
    record["rounds"] = [_entry(f) for f in found]
    mail.atomic_write(res.mbox, series)              # the .mbox first: a record is only there once its file is
    mail.atomic_write(res.yaml, mail.dump_yaml(record))
    return res


def _together(office: mail.Office, root: Path, found: Sequence[Found], sids: Sequence[str],
              res: Result) -> Tuple[str, str]:
    """A series that does not apply on its own may still apply together with the rounds of the other sessions."""
    others = [s for s in office.session_ids() if s not in sids]
    if not others:
        return res.applies, res.detail
    every, _ = find_rounds(office, [*sids, *others])
    if len(every) == len(found) or check_series(root, every)[0] != "yes":
        return res.applies, res.detail
    names = sorted({f.sid for f in every} - set(res.sessions))
    return "together", (f"these rounds build on files that the sessions {', '.join(names)} changed first, so they apply "
                        f"only together with theirs: {res.all_cmd}")


def _rel(path: Path, root: Path) -> str:
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:
        return str(path)


def apply_command(res: Result, root: Path) -> str:
    """What the person runs to commit the series, one commit per round (on a checkout of ``res.head`` if it is known)."""
    if not res.mbox:
        return ""
    return "git am --3way" + (" --keep-cr" if res.keep_cr else "") + f" {_rel(res.mbox, root)}"


def line(res: Result, root: Path) -> str:
    """The one ``[JINX_EXPORT]`` line for the editor to pass on to the person."""
    if not res.mbox or not res.yaml:
        return ""
    out = (f"[JINX_EXPORT] rounds={len(res.rounds)} mbox={_rel(res.mbox, root)} yaml={_rel(res.yaml, root)} "
           f'apply="{apply_command(res, root)}"')
    if res.head:
        out += f" head={res.head[:10]}"
    if res.applies in ("already", "together", "no", "unknown") or res.skipped:
        out += f" check={res.applies}" + (f" skipped={len(res.skipped)}" if res.skipped else "")
    if res.applies == "together":
        out += f' all="{res.all_cmd}"'
    return out


def _touched(f: Found) -> List[str]:
    """The paths a round changes: its record says, and a round without one is read from its diff."""
    named = [str(t) for t in (f.record.get("touches") or [])]
    if named:
        return named
    out: List[str] = []
    for ln in mail.split_mbox(f.data)[1].splitlines():
        if ln.startswith(b"diff --git a/"):
            out.append(ln[len(b"diff --git a/"):].split(b" b/")[0].decode("utf-8", "replace"))
    return out


def uncommitted(root: Path, rounds: Sequence[Found], cap: int = 400) -> List[str]:
    """Paths these rounds change that the working tree has changed too, uncommitted: ``git am`` will not overwrite them.

    That is the normal state right after a session ran in this very tree: its work is in the files already, and the
    series is for another checkout of ``head``. [] when it cannot be told (not a git tree, too many paths).
    """
    paths = sorted({p for f in rounds for p in _touched(f)})
    if not paths or len(paths) > cap:
        return []
    r = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all",
                        "--no-renames", "--", *paths], capture_output=True)
    if r.returncode != 0:
        return []
    return sorted(rec[3:].decode("utf-8", "replace") for rec in r.stdout.split(b"\0") if len(rec) > 3)


# --------------------------------------------------------------------------- command line
def export_main(argv: Sequence[str], agent_dir: Path) -> int:
    """``jinx.py --export [SESSION ...] [--out DIR]``: collect, print what was made, return an exit code."""
    ap = argparse.ArgumentParser(prog="jinx.py", add_help=False)
    ap.add_argument("--export", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args, _ = ap.parse_known_args(list(argv))
    agent_dir = Path(agent_dir)
    ipc = Path(os.environ.get("JINX_IPC_DIR") or (agent_dir / "ipc"))
    root = Path(os.environ.get("JINX_PROJECT_ROOT") or agent_dir.parent).resolve()
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
    named = [s for s in (args.export or []) if s not in ("all", "*")]
    missing = [s for s in named if team.session_dir(office, s) is None]
    if missing:
        print(f"no session {', '.join(missing)} (known: {', '.join(office.session_ids()) or 'none'})")
        return 1
    sids = named or office.session_ids()
    out_dir = Path(args.out).resolve() if args.out else agent_dir / EXPORT_DIR
    try:
        res = build(office, root, sids, out_dir)
    except (OSError, mail.MailError) as e:
        print(f"could not export: {e}")
        return 1
    if not res.rounds and not res.skipped:
        print("no round has changed a file yet" if not named else f"no round of {', '.join(named)} has changed a file yet")
        return 0
    if res.mbox:
        print(line(res, root))
        base = f" (on a checkout of {res.head[:10]})" if res.head else ""
        print(f"commit it, one commit per round: {apply_command(res, root)}{base}")
        if res.applies == "yes":
            print("it applies to HEAD as it is now (checked on a copy of the index; nothing was changed)")
        elif res.applies == "already":
            print("HEAD already holds these changes; the file is for another checkout")
        else:
            print(f"check: {res.detail or res.applies}")
        dirty = uncommitted(root, res.rounds)
        if dirty:
            print(f"note: this working tree already holds uncommitted changes to {len(dirty)} of these files "
                  f"(for example {dirty[0]}), and `git am` does not overwrite those. Apply the series on a clean "
                  f"checkout{' of ' + res.head[:10] if res.head else ''} (another clone, or `git worktree add`), "
                  "or commit this working tree as it is.")
    for s in res.skipped:
        print(f"left out: round {s['round']} of {s['session']}: {s['why']}")
    return 0 if res.clean else 1


def auto(office: mail.Office, root: Path, sid: str, out_dir: Path) -> str:
    """The export of one session at the end of its run: its ``[JINX_EXPORT]`` line, or "" when it changed nothing.

    Never raises: a failed export must not turn a finished run into a failed one.
    """
    try:
        res = build(office, root, [sid], out_dir)
        return line(res, root)
    except Exception as e:                          # noqa: BLE001 - best effort by design
        logger.warning("Could not export the rounds of %s: %s", sid, e)
        return ""
