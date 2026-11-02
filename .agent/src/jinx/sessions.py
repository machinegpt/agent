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
"""Several JINX sessions in one project at the same time.

Without this module every process shares one set of files in ``.agent/`` (the request,
the response, the run state, the state manifest), so two editors that start JINX at once
wipe each other's work. Here each session gets a directory of its own,

    .agent/ipc/sessions/<sid>/   jinx_request.yaml, jinx_response.yaml, jinx_run_state.yaml,
                                 JINX.yaml, mailbox, outbox, a .mbox and a .yaml per round

and the sessions talk to each other through ``mail.py``: they see who else is running, take
exclusive write leases on files, and exchange short YAML messages.

The runner is the only caller. Every function here either returns text for the model or does
bookkeeping; none of them raises for an I/O problem, because coordination must never be what
stops a run that would otherwise succeed (a failure is logged and the work goes on, unprotected).
"""

import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import export, identity, mail, selfpatch, team
from .state import AGENT_DIR

logger = logging.getLogger("jinx.sessions")

# Names of the files the editor reads and writes: unchanged, only their directory differs.
REQUEST_NAME = "jinx_request.yaml"
RESPONSE_NAME = "jinx_response.yaml"
RUN_STATE_NAME = "jinx_run_state.yaml"
MANIFEST_NAME = "JINX.yaml"

# Pseudo-path claimed by whoever edits JINX's own source. The self-patch gate compares the whole
# source tree with one baseline, so two sessions editing it at once would revert each other.
FRAMEWORK_CLAIM = ".agent/src/jinx"

# Shell commands change files without telling anybody which. Around a batch that runs one, the tree is
# compared before and after (git status), and what changed is attributed to the round. The other sessions
# learn from the card whether such a batch is out (``shell``: when it began) and when the last one ended
# (``shell_ended``), which is how they tell whether they overlapped it.
SHELL_STALE_S = 3600                   # a batch that has been "out" this long is not running: its session lost its record
SHELL_UNATTRIBUTED_CAP = 20            # paths named in a round's record as "changed, but not attributable"
SHELL_ADOPT_CAP = 100              # a command that changes more files than this is not an edit (npm install...)
SHELL_APPEND_CAP = 1_000_000       # bytes: larger files are not compared to see whether a change only appended
SHELL_WATCH_BUDGET_S = 2.5         # a tree that takes longer than this to inspect is left alone from then on
SHELL_SCRIPT_KEEP = 8000           # characters of the script kept to see which files it names

# What `git` commands throw away other sessions' uncommitted work. Best effort: it is a pattern, not a sandbox.
_DESTRUCTIVE_GIT = re.compile(
    r"\bgit\b(?:\s+-C\s+\S+)?\s+(?:"
    r"reset\b[^;&|\n]*--hard"
    r"|clean\b[^;&|\n]*(?:\s-[a-zA-Z]*f|\s--force)"
    r"|stash\b(?!\s+(?:list|show)\b)"
    r"|checkout\b[^;&|\n]*(?:\s-f\b|\s--force\b|\s(?:--\s+)?\.(?=\s|$|[;&|]))"
    r"|restore\b[^;&|\n]*\s\.(?=\s|$|[;&|])"
    r")"
)

REFUSE_IPC_WRITE = (
    "%s belongs to JINX (mail, claims and session state) and is not edited by hand. To send a "
    "message, write a draft to %s/<name>.yaml with: to, kind, summary [, needs, re, body]."
)
FRAMEWORK_BUSY = (
    "JINX's own source can only be edited while no other session is running; running now: %s. "
    "Finish the rest of the task first, or change code outside .agent/src/jinx."
)
LOST_WRITE = (
    "%s was written, but session %s took the path over while the write was pending (your lease had "
    "lapsed), so the write may have overwritten their work. Read the file again before relying on it."
)
SHELL_DESTRUCTIVE = (
    "This command would discard uncommitted changes in the whole working tree, and other JINX sessions "
    "are editing it right now (%s). Undo only the files you changed yourself (`git checkout -- <path>`), "
    "or wait until the others have finished."
)
SHELL_CONFLICT = (
    "A shell command of yours changed %s while session %s holds it. That session was told; check the "
    "file before relying on either version."
)
SHELL_FRAMEWORK = (
    "A shell command of yours changed JINX's own source (%s) while other sessions are running (%s). Their "
    "next steps import it, and nothing verifies the change for you: undo it, or make sure it is complete."
)
SEVERAL_SESSIONS = (
    "Several JINX sessions are unfinished and this call cannot be matched to one of them: %s. Resume the one "
    "you started with `python .agent/jinx.py --session <id>`; the id is printed on the [JINX_SESSION] line."
)


class SessionError(Exception):
    """A problem with the way the session was asked for; the message is for the person at the CLI."""


def enabled() -> bool:
    """Multi-session mode is the default; JINX_SESSIONS=0 restores the single shared set of files."""
    return os.environ.get("JINX_SESSIONS", "1").strip().lower() not in ("0", "false", "no", "off")


def ipc_dir() -> Path:
    return Path(os.environ.get("JINX_IPC_DIR") or (AGENT_DIR / "ipc"))


def project_root() -> Path:
    return Path(os.environ.get("JINX_PROJECT_ROOT") or AGENT_DIR.parent).resolve()


def export_dir() -> Path:
    """Where the rounds of a finished run are collected (export.py)."""
    return AGENT_DIR / export.EXPORT_DIR


class Context:
    """The session this process serves: its mail session, its round, and where its files are."""

    def __init__(self, office: mail.Office, session: mail.Session):
        self.office = office
        self.session = session
        self.sid = session.sid
        self.dir = session.dir
        self.root = project_root()
        self.request = self.dir / REQUEST_NAME
        self.response = self.dir / RESPONSE_NAME
        self.run_state = self.dir / RUN_STATE_NAME
        self.manifest = self.dir / MANIFEST_NAME
        self.pending_digest: Optional[mail.Digest] = None
        self.pending_notes: List[str] = []           # outbox problems, reported once next to the digest
        self.briefing: Optional[Dict[str, Any]] = None   # what a session that joined another is told once (team.briefing)
        self.briefing_pending = False                # ... it is in a request that has not been written out yet
        self.resume_cmd = ""                         # what the editor runs to call this session again
        self.others = 0                              # live sessions besides this one when this call began
        self._round: Optional[mail.Round] = None

    @property
    def round(self) -> mail.Round:
        if self._round is None:
            tmp = self.dir / "tmp"
            tmp.mkdir(exist_ok=True)
            self._round = mail.Round(self.session, self.root, tmp)
        return self._round

    def rel(self, path: Path) -> str:
        """``path`` as the editor should write it: relative to the project when it lies inside it."""
        try:
            return Path(os.path.relpath(path, self.root)).as_posix()
        except ValueError:
            return str(path)

    @property
    def outbox_rel(self) -> str:
        return self.rel(self.session.outbox)


def _office() -> mail.Office:
    root = ipc_dir()
    try:
        label = Path(os.path.relpath(root, project_root())).as_posix()
        agent = Path(os.path.relpath(AGENT_DIR, project_root())).as_posix()
    except ValueError:
        label, agent = str(root), str(AGENT_DIR)
    # `cmd` is how the model is told to call JINX again (to look at another session, say).
    return mail.Office(root, label=label, cmd=f"python {agent}/jinx.py")


def _lease() -> int:
    try:
        return max(30, int(float(os.environ.get("JINX_LEASE_S", mail.DEFAULT_LEASE_S))))
    except (ValueError, OverflowError):
        return mail.DEFAULT_LEASE_S


def _active(office: mail.Office) -> List[str]:
    """Sessions that have an unfinished run: their run state is the flag, as in single-session mode."""
    out = []
    for sid in office.session_ids():
        if (office.sessions_dir / sid / RUN_STATE_NAME).exists():
            out.append(sid)
    return out


def _seed(ctx: Context, office: mail.Office) -> None:
    """A manifest a person configured seeds every new session; old finished sessions are tidied away."""
    base = AGENT_DIR / MANIFEST_NAME
    if base.is_file():
        try:
            shutil.copyfile(base, ctx.manifest)
        except OSError as e:
            logger.warning("Could not seed %s from %s: %s", ctx.manifest, base, e)
    office.sweep()


def _unfinished(office: mail.Office) -> str:
    """The unfinished sessions as one line of text for an error: id, round and what each was asked."""
    return ", ".join(f"{s} (r{office.card(s).get('round') or 0}: {(office.card(s).get('task') or '')[:40]})"
                     for s in _active(office))


def _resolve_bare(office: mail.Office, who: List[str]) -> mail.Session:
    """The session of a call that names none: the only unfinished one, or the one this editor has been using.

    The editor is recognised by the fingerprint its earlier calls left on the session (see identity.py). If
    any unfinished session has none, or the caller matches no session or several, nothing is guessed: giving
    one editor another's session would hand it that session's task and files.
    """
    active = _active(office)
    if not active:
        raise SessionError("Cannot start new session without a task description.")
    if len(active) == 1:
        return office.session(active[0])
    marks = {s: [str(t) for t in (office.card(s).get("who") or [])] for s in active}
    if all(marks.values()):
        sid, _ = identity.identify(who, marks)
        if sid:
            return office.session(sid)
    raise SessionError(SEVERAL_SESSIONS % _unfinished(office))


def _join(office: mail.Office, host: str, task: Optional[str], session_id: Optional[str], who: List[str]) -> Context:
    """A new session that joins ``host`` to help: same team, its own files, a briefing about the host."""
    hcard = office.card(host) if mail._SID_RE.match(host) else {}
    if not hcard:
        raise SessionError(f"no session {host!r} to join; unfinished: {_unfinished(office) or 'none'}")
    if hcard.get("state") == "done":
        raise SessionError(f"session {host} has finished; `{office.cmd} --peek {host}` shows what it did. "
                           "Start your own session with a task instead.")
    if session_id and (office.sessions_dir / session_id).exists():
        raise SessionError(f"session {session_id} already exists; leave --session out to start a new one")
    text = " ".join((task or "").split()) or f"Help session {host}: {hcard.get('task') or ''}".strip()
    sess = office.new_session(text, sid=session_id or None, editor=os.environ.get("JINX_EDITOR", ""),
                              lease_s=_lease(), room=hcard.get("room") or host, who=who)
    ctx = Context(office, sess)
    _seed(ctx, office)
    try:
        ctx.briefing = team.briefing(office, project_root(), host, office.cmd)
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Could not prepare the briefing about %s: %s", host, e)
    try:
        sess.send(host, "tell", f"{sess.sid} joined your work to help: {mail._short(text, 100)}")
    except (OSError, mail.MailError) as e:
        logger.warning("Could not tell %s that %s joined: %s", host, sess.sid, e)
    return ctx


def open_session(task: Optional[str], session_id: Optional[str], join: Optional[str] = None) -> Context:
    """Finds or creates the session for this invocation.

    The launch command is the same for every agent. A task starts a new session (``join`` makes it a helper
    in the team of another one). Without a task the session is the one named with ``--session`` (or
    JINX_SESSION); with neither, the session the runtime recognises as this editor's, which is the only
    unfinished one when there is just one.
    """
    try:
        office = _office()
    except (OSError, mail.MailError) as e:
        raise SessionError(f"cannot open {ipc_dir()}: {e}") from e
    who = identity.fingerprint()
    fresh = True
    try:
        if join:
            ctx = _join(office, join, task, session_id, who)
        elif task:
            if session_id and (office.sessions_dir / session_id).exists():
                raise SessionError(
                    f"session {session_id} already exists; resume it without a task, or start a new one "
                    f"by leaving --session out")
            sess = office.new_session(task, sid=session_id or None, editor=os.environ.get("JINX_EDITOR", ""),
                                      lease_s=_lease(), who=who)
            ctx = Context(office, sess)
            _seed(ctx, office)
        else:
            fresh = False
            if session_id:
                try:
                    sess = office.session(session_id)
                except mail.MailError as e:
                    raise SessionError(f"{e}; unfinished sessions: {_unfinished(office) or 'none'}") from e
            else:
                sess = _resolve_bare(office, who)
            ctx = Context(office, sess)
    except mail.MailError as e:
        raise SessionError(str(e)) from e
    return _finish(ctx, who, fresh)


def _finish(ctx: Context, who: List[str], fresh: bool) -> Context:
    """Remember which editor this session belongs to, and work out what the editor should call next time."""
    explicit = f"{ctx.office.cmd} --session {ctx.sid}"
    ctx.resume_cmd = explicit
    try:
        if not fresh:
            card = ctx.session.card()
            old, n = [str(t) for t in (card.get("who") or [])], int(card.get("who_n") or 0)
            tokens, seen = identity.refine(old, who, n)
            if tokens != old or seen != n:
                ctx.session.beat(who=tokens, who_n=seen)
        ctx.resume_cmd = _resume_command(ctx)
        ctx.others = len(ctx.office.peers(exclude=ctx.sid))
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Could not note which editor session %s belongs to: %s", ctx.sid, e)
    return ctx


def _resume_command(ctx: Context) -> str:
    """The command the editor is told to run next: bare when the runtime will know it again, else with the id.

    Bare only when this session's marks have survived two calls (so the marks of the shell that ran one
    command are gone), every unfinished session has marks, and they single this session out. Anything less
    and the editor is given the id, which always works.
    """
    explicit = f"{ctx.office.cmd} --session {ctx.sid}"
    card = ctx.session.card()
    mine = [str(t) for t in (card.get("who") or [])]
    if int(card.get("who_n") or 0) < 2 or not mine:
        return explicit
    marks = {s: [str(t) for t in (ctx.office.card(s).get("who") or [])] for s in _active(ctx.office)}
    marks[ctx.sid] = mine
    if not all(marks.values()):
        return explicit
    sid, _ = identity.identify(mine, marks)
    return ctx.office.cmd if sid == ctx.sid else explicit


def describe(ctx: Context) -> str:
    """The one line every invocation prints, so the editor always has the exact paths, and how many others run."""
    resume = ctx.resume_cmd or f"python {ctx.rel(AGENT_DIR)}/jinx.py --session {ctx.sid}"
    line = '[JINX_SESSION] id=%s request=%s response=%s resume="%s"' % (
        ctx.sid, ctx.rel(ctx.request), ctx.rel(ctx.response), resume)
    return line + (f" others={ctx.others}" if ctx.others else "")


def heartbeat(ctx: Context, rnd: Optional[int] = None) -> None:
    try:
        fields: Dict[str, Any] = {"state": "working"}
        if rnd is not None:
            fields["round"] = rnd
        ctx.session.beat(**fields)
    except OSError as e:
        logger.warning("Could not refresh the session heartbeat: %s", e)


def alone(ctx: Context) -> bool:
    try:
        return ctx.session.alone()
    except OSError:
        return True


def self_patch_allowed(ctx: Context) -> bool:
    """The self-patch gate may act when nobody else is running, or when this session holds the source."""
    try:
        return ctx.session.alone() or ctx.session.holds_claim(FRAMEWORK_CLAIM)
    except OSError:
        return True


# ------------------------------------------------------------------------------ mail in the request
RECENT_READS = 12          # of the files the model has read, this many (the latest) count as the ones it works with


def working_set(ctx: Context) -> List[str]:
    """The files this session is busy with: held now, changed lately, and read most recently.

    The board tells the model about other sessions that touch these, and stays quiet about the rest.
    """
    card = ctx.session.card()
    rnd = ctx.round
    files = list((card.get("holds") or {}).keys()) + list(rnd.base) + list(mail.touched_of(card))
    files += list(rnd.seen)[-RECENT_READS:]
    return list(dict.fromkeys(files))


def digest_message(ctx: Context, round_start: bool = True) -> Optional[Dict[str, Any]]:
    """The message that tells the model who else is running, what they do and what mail it has, or None.

    None is the common case for a session that runs alone: no tokens are spent on this feature.
    The digest is remembered in ``ctx.pending_digest`` and committed by :func:`commit_digest`
    once the request that carries it has been written. ``round_start`` is False for the requests
    inside a round: they carry mail and news about the files the session works with, and leave the
    whole list of the other sessions for the start of a round. A session that joined another one is
    given the briefing about it once, in front of everything else.
    """
    try:
        for item in ctx.session.flush_outbox():
            _note_outbox(ctx, *item)
        d = ctx.session.digest(round_start=round_start, working=lambda: working_set(ctx))
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Could not build the mail digest: %s", e)
        d = None
    extra = ctx.pending_notes
    if d is None and not extra and not ctx.briefing:
        return None
    ctx.pending_digest = d
    parts: List[str] = []
    if ctx.briefing:
        parts.append(mail.dump_yaml({"join": ctx.briefing}))
        ctx.briefing_pending = True
    if d is not None:
        parts.append(d.text)
    if extra:
        parts.append(mail.dump_yaml({"outbox": list(extra)}))
        ctx.pending_notes = []
    return {"role": "user", "content": "".join(parts)}


def commit_digest(ctx: Context) -> None:
    d, ctx.pending_digest = ctx.pending_digest, None
    if ctx.briefing_pending:
        ctx.briefing, ctx.briefing_pending = None, False          # told once: it is in a request that was written
    if d is None:
        return
    try:
        ctx.session.commit_digest(d)
    except (OSError, mail.MailError) as e:
        logger.warning("Could not mark the mail as shown: %s", e)


def _note_outbox(ctx: Context, name: str, ok: bool, why: str) -> None:
    """A draft that was refused is reported to the model once, with the reason, next to the digest."""
    if not ok:
        ctx.pending_notes.append(f"{name} was not sent: {why}")


def flush_outbox(ctx: Context) -> None:
    try:
        for item in ctx.session.flush_outbox():
            _note_outbox(ctx, *item)
    except (OSError, mail.MailError) as e:
        logger.warning("Could not deliver the outbox: %s", e)


# ------------------------------------------------------------------------------ files: claims
def project_rel(ctx: Context, raw: str) -> Optional[str]:
    """``raw`` relative to the project root, or None when it lies outside the project."""
    try:
        p = Path(raw)
        if not p.is_absolute():
            p = ctx.root / p
        return Path(os.path.realpath(p)).relative_to(ctx.root).as_posix()
    except (ValueError, OSError):
        return None


def _in_ipc(ctx: Context, rel: str) -> bool:
    ipc = ctx.rel(ipc_dir())
    return rel == ipc or rel.startswith(ipc + "/")


def screen_call(ctx: Context, name: str, params: Dict[str, Any]) -> Optional[str]:
    """Checks one file_read / file_write before it is dispatched. Returns a refusal for the model, or None.

    A read is only noted: it is the version the model is about to see. A write needs the exclusive
    lease on its path, and is refused when the file changed after the model read it, because
    file_write replaces the whole file and would silently discard that change.
    """
    raw = params.get("path")
    if name not in ("file_read", "file_write") or not isinstance(raw, str) or not raw.strip():
        return None
    rel = project_rel(ctx, raw)
    if rel is None:
        return None                                   # outside the project: nobody else is coordinated there
    try:
        if _in_ipc(ctx, rel):
            if name == "file_read":
                return None
            outbox = ctx.outbox_rel
            if Path(rel).parent.as_posix() == outbox and rel.endswith(".yaml"):
                return None                           # a mail draft: the one thing the model may write here
            return REFUSE_IPC_WRITE % (rel, outbox)
        if name == "file_read":
            ctx.round.note_read(rel)
            return None
        if selfpatch.within_src_tree(raw):
            busy = _framework_refusal(ctx)
            if busy:
                return busy
        res = ctx.round.claim_for_write(rel)
        return None if res.ok else _refusal(ctx, res)
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Coordination failed for %s %s (continuing unprotected): %s", name, raw, e)
        return None


def _refusal(ctx: Context, res: mail.ClaimResult) -> str:
    """Why a write was refused, and for a file somebody else holds, how to reach the holder and see its work."""
    text = res.explain()
    if res.holder and res.reason != "stale":
        text += (f" (message: file_write {ctx.outbox_rel}/<name>.yaml with to: {res.holder}, kind: ask, summary: ...; "
                 f"its work: {ctx.office.cmd} --peek {res.holder})")
    return text


def _framework_refusal(ctx: Context) -> Optional[str]:
    """Editing JINX's own source is for a session that has it to itself; None means go ahead."""
    if ctx.session.holds_claim(FRAMEWORK_CLAIM):
        return None
    others = [p["id"] for p in ctx.office.peers(exclude=ctx.sid)]
    if others:
        return FRAMEWORK_BUSY % ", ".join(others)
    res = ctx.session.claim(FRAMEWORK_CLAIM)
    if not res.ok:
        return res.explain()
    _ensure_baseline()
    return None


def _ensure_baseline() -> None:
    """Make sure a self-patch baseline exists before the edit that is about to be dispatched.

    A session that started while others were running did not take one (see the runner), and the
    previous owner may have dropped it when it finished. The gate verifies against it and
    restores from it, so an edit made without one would go unchecked.
    """
    if not selfpatch.SELF_PATCH_ENABLED or selfpatch.BASELINE_DIR.exists():
        return
    try:
        with selfpatch.gate_lock() as held:
            if held and not selfpatch.BASELINE_DIR.exists():
                selfpatch.capture_baseline()
    except OSError as e:
        logger.warning("Could not capture the self-patch baseline: %s", e)


def screen_shell(ctx: Context, script: Any) -> Optional[str]:
    """Refuses a shell command that would wipe the other sessions' uncommitted work; None lets it through."""
    if not isinstance(script, str) or not _DESTRUCTIVE_GIT.search(script):
        return None
    try:
        others = [p["id"] for p in ctx.office.peers(exclude=ctx.sid)]
    except OSError:
        return None
    return SHELL_DESTRUCTIVE % ", ".join(others) if others else None


def _own_prefixes(ctx: Context) -> List[str]:
    """Paths of JINX's own bookkeeping: never part of anybody's round."""
    agent = ctx.rel(AGENT_DIR)
    return [f"{agent}/ipc/", f"{agent}/{export.EXPORT_DIR}/", f"{agent}/notices/", f"{agent}/.selfpatch",
            f"{agent}/lessons.yaml", f"{agent}/JINX.yaml", f"{agent}/jinx_"]


def before_batch(ctx: Context, calls: List[Dict[str, Any]]) -> None:
    """A batch with a shell command is about to go to the editor: remember what the tree looks like now.

    Files are changed by shell commands (sed, a Python script, a formatter) as often as by file_write, and
    only file_write says which file. So the tree is compared before and after the batch (see
    :func:`after_batch`) and the difference is attributed to this round. Skipped when the project is not
    a git working tree, or when looking at it is slow (a huge tree is not worth a delay on every batch).
    """
    scripts = [str((c.get("params") or {}).get("script") or "") for c in calls if c.get("name") == "bash_exec"]
    if not scripts:
        return
    started = time.time()
    try:
        if alone(ctx):
            _ensure_baseline()                # a framework edit by a shell command must be verifiable too
        rnd = ctx.round
        rnd.begin()                           # the commit this round starts from: the command may commit
        if not rnd.git or not rnd.watch or rnd.shell:
            return                            # (a record that exists is the earliest state: this is a retry)
        t0 = time.monotonic()
        pre = mail.dirty_files(ctx.root, _own_prefixes(ctx))
        if time.monotonic() - t0 > SHELL_WATCH_BUDGET_S:
            rnd.watch = False
            logger.warning("Inspecting the working tree took %.1fs: shell changes are no longer attributed.",
                           time.monotonic() - t0)
        rnd.shell = {
            "started": started,
            "dirty": {p: (list(e) if e else None) for p, e in pre.items()},
            "script": "\n".join(scripts)[:SHELL_SCRIPT_KEEP],
        }
        rnd._save()
        ctx.session.beat(shell=started)       # the others see that a batch with a shell command is out
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Could not look at the tree before the shell command (continuing unprotected): %s", e)


def _mentions(script: str, rel: str) -> bool:
    """Whether a shell script names this path, whole or just its file name, as a word of its own.

    ("a.txt" is not named by "data.txt" or "a.txt.bak"; "bash/a.txt" is named by "./bash/a.txt".)
    """
    for name in {rel, Path(rel).name}:
        if name and re.search(r"(?<![\w.\-])" + re.escape(name) + r"(?![\w\-]|\.\w)", script):
            return True
    return False


def _only_appended(ctx: Context, rel: str, before: mail.Entry) -> bool:
    """Whether the file is now what it was before the command plus text added at its end.

    Lines appended to a file somebody else holds merge by themselves, so they are no conflict: nobody is
    interrupted. What protects the holder is the stale-read guard, which refuses its next whole-file write
    until it has read the file again.
    """
    if before is None:
        return False
    try:
        full = ctx.root / rel
        if not full.is_file() or full.is_symlink() or full.stat().st_size > SHELL_APPEND_CAP:
            return False
        old = mail.blob_bytes(ctx.root, before[1], SHELL_APPEND_CAP)
        if old is None:
            return False
        cur = full.read_bytes()
    except (OSError, mail.MailError):
        return False
    old_norm = old.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    cur_norm = cur.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return len(cur_norm) > len(old_norm) and cur_norm.startswith(old_norm)


def _shell_overlap(ctx: Context, started: float) -> bool:
    """Whether another session may have been running shell commands while this session's batch was out.

    A command that changes files without naming them (`sed -i ... *.py`, a formatter, a generator) leaves
    the tree different, and the before/after comparison cannot tell whose command did it when two were
    running at once. Cautious: anything that cannot be read counts as an overlap.
    """
    now = time.time()
    for sid in ctx.office.session_ids():
        if sid == ctx.sid:
            continue
        card = ctx.office.card(sid)
        if not card:
            return True
        try:
            out = float(card.get("shell") or 0.0)
            ended = float(card.get("shell_ended") or 0.0)
        except (TypeError, ValueError):
            return True
        if out and now - out < SHELL_STALE_S and ctx.office.alive(sid, card):
            return True                       # its batch is out with its editor right now
        if ended >= started:
            return True                       # it finished a batch after ours began
    return False


def after_batch(ctx: Context) -> List[str]:
    """The batch is done: give the files a shell command changed to this round. Returns notes for the model.

    A file that changed and that nobody else holds is leased to this session now and enters the round's
    patch with its state from before the batch as the baseline. A file another session holds is not ours to
    take: if the command names it, both sessions are told. Normal cases say nothing, so cost no tokens.

    The comparison sees every change in the tree, whoever made it. While another session was running shell
    commands too, a change is taken only when this session's own command names the file; the rest is
    listed in the round's record as not attributable instead of being credited to whichever session looks first.
    """
    if not ctx.session.card().get("shell"):
        return []                             # before_batch started no watch: nothing to attribute (and no git to spawn)
    rnd = ctx.round
    if not rnd.shell:
        # The record is gone (round.yaml was lost): do not leave the others waiting for a batch that is over.
        ctx.session.beat(shell=None, shell_ended=time.time())
        return []
    notes: List[str] = []
    try:
        data = rnd.shell
        pre: Dict[str, mail.Entry] = {p: (tuple(e) if e else None) for p, e in (data.get("dirty") or {}).items()}
        script = str(data.get("script") or "")
        try:
            started = float(data.get("started") or 0.0)
        except (TypeError, ValueError):
            started = 0.0
        rnd.shell = None                      # consumed, whatever happens below
        rnd._save()
        # One write: the batch stops counting as running at the moment its end is stamped, so there is no gap for the others.
        ctx.session.beat(shell=None, shell_ended=time.time())
        post = mail.dirty_files(ctx.root, _own_prefixes(ctx), store=False)
        changed: List[Tuple[str, mail.Entry]] = []
        for p in sorted(set(pre) | set(post)):
            before = pre[p] if p in pre else mail.head_entry(ctx.root, p)      # not dirty before: as committed
            after = post[p] if p in post else mail.head_entry(ctx.root, p)     # not dirty now: as committed
            if before != after:
                changed.append((p, before))
        if not changed:
            return []
        if len(changed) > SHELL_ADOPT_CAP:
            rnd.notes["shell_unrecorded"] = len(changed)       # said in the round's record: it is incomplete
            rnd._save()
            return []
        peers = [x["id"] for x in ctx.office.peers(exclude=ctx.sid)]
        overlap = bool(peers) and _shell_overlap(ctx, started)
        unattributed: List[str] = []
        for p, before in changed:
            if p in rnd.base:
                continue
            named = _mentions(script, p)
            info = ctx.session.claim_info(p) or {}
            holder = str(info.get("owner") or "") if info.get("state") == "held" else ""
            foreign = bool(holder) and holder != ctx.sid and ctx.office.alive(holder)
            if not foreign and overlap and not named:
                unattributed.append(p)        # it may as well be the other session's command
                continue
            ours = False
            if not foreign:
                # While another session's command was running, the file may already hold its unrecorded change:
                # start from the committed state then, so that change lands in some patch instead of in none.
                got = rnd.adopt(p, before, mail.head_entry(ctx.root, p)) if overlap else rnd.adopt(p, before)
                ours = got == "adopted"
                if got == "busy":             # somebody took the lease between the look and the claim
                    holder = str((ctx.session.claim_info(p) or {}).get("owner") or "")
                    foreign = bool(holder) and holder != ctx.sid
            if not ours and foreign and named and not _only_appended(ctx, p, before):
                notes.append(SHELL_CONFLICT % (p, holder))
                try:
                    ctx.session.send(holder, "conflict", f"{p}: a shell command of mine changed it while you hold it",
                                     body={"path": p}, needs="reply")
                except (mail.MailError, OSError):
                    pass
            if (ours or named) and peers and selfpatch.within_src_tree(str(ctx.root / p)):
                notes.append(SHELL_FRAMEWORK % (p, ", ".join(peers)))
        if unattributed:
            kept = sorted(set(rnd.notes.get("shell_unattributed") or []) | set(unattributed))
            rnd.notes["shell_unattributed"] = kept[:SHELL_UNATTRIBUTED_CAP]
            rnd._save()
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Could not attribute the shell changes (continuing): %s", e)
    return notes


def check_writes(ctx: Context, calls: List[Dict[str, Any]]) -> List[str]:
    """After a tool batch: which writes landed after the session had lost its lease on the path.

    The check before the write and this one bracket it. Generations only ever grow, so if the lease
    is still this session's now, nobody else can have held the path while the write happened.
    """
    notes: List[str] = []
    try:
        for c in calls:
            params = c.get("params") if isinstance(c.get("params"), dict) else {}
            raw = params.get("path")
            if c.get("name") != "file_write" or not isinstance(raw, str):
                continue
            rel = project_rel(ctx, raw)
            if rel is None or _in_ipc(ctx, rel) or rel not in ctx.round.base:
                continue
            if ctx.session.holds_claim(rel):
                ctx.round.note_written(rel)           # the model knows what it has just written
                continue
            info = ctx.session.claim_info(rel) or {}
            holder = str(info.get("owner") or "another session")
            notes.append(LOST_WRITE % (rel, holder))
            if holder.startswith("s"):
                try:
                    ctx.session.send(holder, "conflict", f"{rel}: my write may have overwritten yours",
                                     body={"path": rel}, needs="reply")
                except (mail.MailError, OSError):
                    pass
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Post-write check failed: %s", e)
    return notes


# ------------------------------------------------------------------------------ rounds and the end
def round_closed(ctx: Context, rnd: int, summary: str, evidence: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """The model's round ended: record it as rounds/NNNN.mbox + NNNN.yaml, release the file leases, update the card.

    ``evidence`` is what the model said about the round (see runner._round_evidence); it goes into the .yaml.
    """
    out: Optional[Path] = None
    ctx.round.last_touched = []
    summary, evidence = mail.clean_text(summary), mail.clean_text(evidence)     # a lone surrogate cannot be written
    try:
        out = ctx.round.close(rnd, summary, evidence)
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Could not record round %s: %s", rnd, e)
        try:
            ctx.session.release_all()
        except OSError:
            pass
    _publish_round(ctx, rnd, summary)
    return out


def _publish_round(ctx: Context, rnd: int, summary: str) -> None:
    """Put on the card what the others see of this round: what it tried, and which files it changed."""
    try:
        card = ctx.session.card()
        touched = mail.touched_of(card)
        for p in ctx.round.last_touched:
            touched.pop(p, None)
            touched[p] = rnd                       # re-inserted last: the dict is ordered by recency
        while len(touched) > mail.MAX_TOUCHED:
            touched.pop(next(iter(touched)))
        fields: Dict[str, Any] = {"touched": touched}
        if summary:
            fields["doing"] = mail._short(summary, 100)
        ctx.session.beat(**fields)
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Could not update the session card after round %s: %s", rnd, e)


def end(ctx: Context, *, completed: bool = False, status: str = "exit") -> str:
    """The session is over (finished, deadlocked, aborted): free everything it held and say goodbye.

    ``completed`` is True for a run that ended on its own terms (``status``: exit, deadlock,
    hard_cap); its team (the host it joined, the helpers that joined it) is told so, with a pointer to the
    .mbox of its last rounds. Everybody else sees it leave the board (``gone``) at their next round
    start: the notice would cost them tokens for a session they do not work with. Any other end (an
    error, a signal) just frees the leases and marks the session done.

    A completed run that changed files also has its rounds collected into one .mbox and .yaml
    (export.py). What comes back is the ``[JINX_EXPORT]`` line for the editor to pass on, or "".
    """
    try:
        for item in ctx.session.flush_outbox():     # drafts written in the last round still go out
            _note_outbox(ctx, *item)
    except (OSError, mail.MailError) as e:
        logger.warning("Could not deliver the last drafts: %s", e)
    try:
        if completed and ctx.office.teammates(ctx.session.sid):
            card = ctx.session.card()
            changed = sorted(ctx.session.rounds.glob("*.mbox"))
            refs = [mail.patch_ref(ctx.office, p) for p in changed[-5:]]
            task = mail._short(card.get("task"), 60)
            body: Dict[str, Any] = {"status": status, "rounds": card.get("round"), "changed_rounds": len(changed)}
            ctx.session.send("room", "tell", f"{status}: {task}", body=body, refs=refs)
    except (OSError, ValueError, mail.MailError) as e:
        logger.warning("Could not announce the end of the session: %s", e)
    exported = ""
    if completed:
        exported = export.auto(ctx.office, ctx.root, ctx.session.sid, export_dir())
    try:
        ctx.session.close()
    except OSError as e:
        logger.warning("Could not close the session cleanly: %s", e)
    return exported
