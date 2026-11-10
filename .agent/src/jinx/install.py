# Copyright 2026 JINX Enterprise Team. All Rights Reserved.
#
# Use of this software is governed by the JINX End User License Agreement (EULA): the LICENSE.md file
# at the root of the JINX repository (https://github.com/machinegpt/agent), and in the `.agent` folder of
# a project where JINX is installed. It is free for personal, educational and non-commercial use,
# including non-commercial research; commercial use needs a commercial license (section 2 of the EULA).
# ==============================================================================
"""Installs JINX into a project: ``jinx install`` and ``jinx doctor``.

    pip install git+https://github.com/machinegpt/agent      # 1. the `jinx` command
    jinx install                                              # 2. in the project folder: set JINX up there

``install`` puts the runtime into ``.agent/`` (``jinx.py`` and ``src/jinx/``, and with them the text of the license, ``LICENSE.md``
and its translations: a copy of JINX does not go anywhere without it), the JINX rules into the instruction file of
every editor (``CLAUDE.md`` for Claude Code, ``AGENTS.md`` for Codex CLI, OpenCode, Grok Build and Kimi Code CLI, ``GEMINI.md``
for Gemini CLI), the ``/jinx`` command into the skill folders the editors read (``.claude/skills/jinx/`` and ``.agents/skills/jinx/``) and the
lines that keep the runtime state out of git into ``.gitignore``, then runs ``doctor``. Run it again after
``pip install -U`` and it brings the runtime up to date: a file that was changed since it was installed (by a person, or
by JINX improving its own source) is left alone, and the new version is written beside it as ``<name>.new``.

The instruction files and .gitignore are not owned by JINX: only the text between its markers is ever written, and a
project's own text around it is kept as it is, line endings included. Nothing here needs more than the standard library.

``/jinx`` is a skill: the model of the editor calls it by name and gets back what JINX printed. In Claude Code the skill
runs the runtime itself (the shell command is in the skill, not in the rules, and not in what the model reads); the other
editors have no way to run a command for a skill, so theirs tells the model which command to run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PKG = Path(__file__).resolve().parent               # the jinx package this code belongs to
AGENT_DIR = ".agent"
MANIFEST = ".installed.json"                        # under .agent/: what an install wrote, to tell a changed file from an untouched one
BASELINE = ".selfpatch_baseline"                    # under .agent/: the source a run is judged against (selfpatch.BASELINE_DIR)
# The text a project gets in its instruction files / .gitignore is the text between these markers in the repository's own files.
_RULES = re.compile(r"<!--\s*JINX:BEGIN\b.*?-->.*?<!--\s*JINX:END\s*-->", re.S)
_IGNORE = re.compile(r"# >>> JINX >>>.*?# <<< JINX <<<", re.S)
_LEGACY_RULES = re.compile(r"route absolutely every single user message")     # the rules of a copy made without `jinx install`
_VERSION = re.compile(r"""__version__\s*=\s*["']([^"']+)["']""")
_PYTHON_MIN = (3, 10)

# The instruction file each editor reads (always, for the whole session), and the editors that read it. OpenCode reads
# AGENTS.md and falls back to CLAUDE.md only where there is no AGENTS.md; Grok Build reads both files; Kimi Code CLI reads
# AGENTS.md and no other of the three.
INSTRUCTION_FILES = (("CLAUDE.md", "Claude Code"),
                     ("AGENTS.md", "Codex CLI, OpenCode, Grok Build and Kimi Code CLI"),
                     ("GEMINI.md", "Gemini CLI"))
# The skill that is `/jinx`, in the folder each group of editors reads. A skill is a SKILL.md that the model of the editor
# loads by its name when the rules (or the person) tell it to. OpenCode reads both folders and takes the second; so does
# Grok Build (its search tries `.agents` before `.claude`, and the first skill of a name is the one it keeps). Kimi Code CLI
# reads `.agents/skills` (and its own `.kimi-code/skills`), not `.claude/skills`.
CLAUDE_SKILL = ".claude/skills/jinx/SKILL.md"
AGENTS_SKILL = ".agents/skills/jinx/SKILL.md"
SKILLS = ((CLAUDE_SKILL, "Claude Code"), (AGENTS_SKILL, "Gemini CLI, Codex CLI, OpenCode, Grok Build and Kimi Code CLI"))
# Where the release's text is left when the file in the project is the person's by now (a file of the project is never replaced then).
_NEW_NAME = {".gitignore": "gitignore.new", CLAUDE_SKILL: "jinx-skill-claude.md.new", AGENTS_SKILL: "jinx-skill-agents.md.new"}
_EDITORS = ("claude", "codex", "opencode", "gemini", "grok", "kimi")        # how the editors are started
# The text of the license, English first (it is the one that governs). A project that is given the runtime is given it with this.
LICENSE_FILES = ("LICENSE.md", "LICENSE_RU.md", "LICENSE_ZH.md")


class InstallError(Exception):
    """Something the person has to fix; the message says what."""


class Kit:
    """What one JINX release hands a project: the runtime, the text of the license, the rules for the instruction files and
    the lines for .gitignore."""

    def __init__(self, entry: Path, rules: Path, ignore: Path, package: Path, licenses: Optional[Dict[str, Path]] = None):
        """``entry``: jinx.py. ``rules``: the file the rules are taken from. ``ignore``: the same for the .gitignore lines.
        ``package``: the folder of the jinx package. ``licenses``: the files of the license text, by name."""
        self.entry, self.rules, self.ignore, self.package = entry, rules, ignore, package
        self.licenses: Dict[str, Path] = licenses or {}

    @property
    def version(self) -> str:
        """The version of this release (``unknown`` where it cannot be read)."""
        return read_version(self.package / "__init__.py") or "unknown"


def _licenses(folder: Path) -> Dict[str, Path]:
    """The files of the license text that are in ``folder``, by name."""
    return {name: folder / name for name in LICENSE_FILES if (folder / name).is_file()}


def find_kit() -> Kit:
    """The kit of this copy of JINX: packed next to the package by the build (``pip install``), or the checkout itself."""
    packed = PKG / "_kit"
    if (packed / "jinx.py").is_file():
        kit = Kit(packed / "jinx.py", packed / "CLAUDE.md", packed / "gitignore", PKG, _licenses(packed))
    else:
        agent = PKG.parent.parent                   # a checkout: .agent/src/jinx
        root = agent.parent
        kit = Kit(agent / "jinx.py", root / "CLAUDE.md", root / ".gitignore", PKG, _licenses(root))
    missing = [p.name for p in (kit.entry, kit.rules, kit.ignore) if not p.is_file()]
    if LICENSE_FILES[0] not in kit.licenses:
        missing.append(LICENSE_FILES[0])
    if missing:
        raise InstallError("this copy of JINX has no " + ", ".join(missing) + " to install from: "
                           "install the whole repository (pip install git+https://github.com/machinegpt/agent)")
    return kit


def read_version(init: Path) -> Optional[str]:
    """The ``__version__`` that ``init`` (the ``__init__.py`` of a jinx package) declares; None where it cannot be read."""
    try:
        m = _VERSION.search(init.read_text(encoding="utf-8"))
    except OSError:
        return None
    return m.group(1) if m else None


# --------------------------------------------------------------------------- the /jinx skill
_SKILL_DESCRIPTION = ("Runs a request through JINX. Use for every user message, greeting or task, "
                      "and to go on with a JINX run in progress.")


def claude_skill(python: str = "python", shell: str = "bash") -> str:
    """The skill of Claude Code. Its body is one shell command that Claude Code runs *before* the model reads the skill, so
    the model never sees the command, only what JINX printed; and it hands the arguments over in one piece (a here-document,
    or a here-string in PowerShell) so that no quoting of the person's message is ever needed. See jinx.py (``--via``).

    The message goes into the script as it is, so a line of it that is exactly ``JINX_ARGS`` (in PowerShell: one that begins
    with ``'@``) ends the here-document early, and what follows it would be read as commands. No delimiter prevents that:
    Claude Code fills ``${CLAUDE_SESSION_ID}`` into the person's text too, so a delimiter made of it can be written by the
    message (tried on Claude Code 2.1), and ``exec python ...`` (which would end the script at the first command) is refused
    by its permission check. What stops it is that check: ``allowed-tools`` lets only ``<python> .agent/jinx.py *`` through,
    and any other command in the script has to be approved (where nobody can, the skill does not run)."""
    head = ("---\nname: jinx\ndescription: %s\nargument-hint: '<request> | --session <id>'\n" % _SKILL_DESCRIPTION)
    if shell == "powershell":
        return (head + "shell: powershell\nallowed-tools: PowerShell(%s .agent/jinx.py *)\n---\n" % python
                + "```!\n%s .agent/jinx.py --via /jinx --args @'\n$ARGUMENTS\n'@\n```\n" % python)
    return (head + "allowed-tools: Bash(%s .agent/jinx.py *)\n---\n" % python
            + "```!\n%s .agent/jinx.py --via /jinx --stdin <<'JINX_ARGS'\n$ARGUMENTS\nJINX_ARGS\n```\n" % python)


def agents_skill(python: str = "python") -> str:
    """The skill of the other editors (Gemini CLI, Codex CLI, OpenCode, Grok Build, Kimi Code CLI): they load its text into the
    conversation and have no way to run a command for it, so it says which command the model is to run, and the model runs it
    with its shell tool."""
    return ("---\nname: jinx\ndescription: %s\n---\n" % _SKILL_DESCRIPTION
            + "`/jinx <arguments>` means running this in the project folder, with your shell tool:\n\n"
              "    %s .agent/jinx.py <arguments>\n\n" % python
            + "The arguments are options of JINX (`--session <id>`, `--peek <id>`, `--join <id>`, `--export`: pass them as they "
              "are) or the user's message. Give the message as ONE argument, exactly as written: in single quotes, so that the "
              "shell reads nothing in it (a `'` inside it is written `'\\''` in bash and `''` in PowerShell), and after `--` "
              "when it begins with `-`. What the command prints is what `/jinx` "
              "returns: its `[JINX_SESSION]` line has the `resume` command, which you run again and again until the run ends. "
              "Where the rules of the project say `[JINX_EXIT] <code>`, that is the exit code of this command when it is not 0 "
              "(1: the launch or the run failed; 2: the run reached its limit of rounds). The instruction file of the project "
              "says what to do with what JINX prints.\n")


def skill_text(name: str, python: str, shell: str) -> str:
    """The text this release writes to the skill file ``name`` (``CLAUDE_SKILL`` or ``AGENTS_SKILL``)."""
    return claude_skill(python, shell) if name == CLAUDE_SKILL else agents_skill(python)


def default_shell() -> str:
    """The shell Claude Code runs the skill's command in: PowerShell on Windows (where it is there without Git Bash), else bash."""
    return "powershell" if os.name == "nt" else "bash"


def _skill_shell(text: str) -> str:
    """The shell a skill of Claude Code is written for: the one its ``shell:`` line names, else bash."""
    return "powershell" if re.search(r"^shell:\s*powershell\s*$", text, re.M) else "bash"


def _chosen_shell(root: Path) -> str:
    """The shell the skill of Claude Code in the project is written for, so that `--claude-shell` is asked for once, not on every update."""
    try:
        return _skill_shell((root / CLAUDE_SKILL).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return default_shell()


# --------------------------------------------------------------------------- files
def digest(data: bytes) -> str:
    """Line endings do not count: a checkout that converts them (``core.autocrlf``) has not changed a file."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _write(path: Path, data: bytes) -> None:
    """Writes ``data`` to ``path`` whole or not at all: into a temporary file beside it, which then takes its place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".jinx-tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def runtime_files(kit: Kit) -> Dict[str, bytes]:
    """{path under .agent/: content} of the runtime of this release, and of the text of the license that goes with it."""
    files = {"jinx.py": kit.entry.read_bytes()}
    for p in sorted(kit.package.glob("*.py")):
        files[f"src/jinx/{p.name}"] = p.read_bytes()
    for name, path in sorted(kit.licenses.items()):
        files[name] = path.read_bytes()
    return files


def _save_manifest(agent: Path, version: str, files: Dict[str, str], blocks: Dict[str, str]) -> None:
    """Records what an install wrote, for the next one (see ``_load_manifest``)."""
    _write(agent / MANIFEST, (json.dumps({"jinx": version, "files": files, "blocks": blocks}, indent=2,
                                         sort_keys=True) + "\n").encode("utf-8"))


def _load_manifest(agent: Path) -> Dict[str, Dict[str, str]]:
    """What the last install wrote: {"files": {path under .agent/: digest}, "blocks": {file of the project: digest}}.

    "blocks" holds both kinds of file of the project: the ones JINX puts a block into (instruction files, .gitignore)
    and the ones that are JINX's from the first line to the last (the skills).
    """
    out: Dict[str, Dict[str, str]] = {"files": {}, "blocks": {}}
    try:
        data = json.loads((agent / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for key in out:
        part = data.get(key) if isinstance(data, dict) else None
        if isinstance(part, dict):
            out[key] = {str(k): str(v) for k, v in part.items()}
    # The manifest is a file of the project, and a project is not always one's own. An install deletes the files
    # it lists that a release no longer has, so a path that leads out of .agent/ (``../..``, absolute) is never
    # one that it wrote: it is dropped here, before anything is done with it.
    out["files"] = {rel: digest_ for rel, digest_ in out["files"].items() if _stays_under_agent(rel)}
    return out


def _stays_under_agent(rel: str) -> bool:
    """``rel`` is a relative path that stays below the folder it is taken from: no ``..``, no root, no drive."""
    path = Path(rel.replace("\\", "/"))
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts and ":" not in path.parts[0]


def _active_runs(agent: Path) -> List[str]:
    """Runs that have not ended: the run state is their flag. Their source is not to change under them.

    So is a self-patch baseline: it is dropped when a run ends cleanly and left when one died with an edit of JINX's
    own source in flight, for the next start to judge. A runtime replaced under it would be rolled back to it.
    """
    found = [p for p in [agent / "jinx_run_state.yaml", *sorted((agent / "ipc" / "sessions").glob("*/jinx_run_state.yaml"))]
             if p.is_file()]
    names = [p.parent.name if p.parent != agent else "the shared run" for p in found]
    if (agent / BASELINE).exists():
        names.append("a self-patch check that did not finish")
    return names


def install_runtime(agent: Path, files: Dict[str, bytes], known: Dict[str, str], force: bool = False,
                    record: Optional[Dict[str, str]] = None) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Writes the runtime into ``agent``. Returns (what happened, path) per file (added, updated, same, kept, removed)
    and the digests to record.

    A file that differs from the release is replaced only when it is what the last install wrote (nobody changed
    it since) or with ``force``; otherwise it stays, and the release's version is written beside it as ``.new``.
    The digests go into ``record`` as the files are written (the dict is handed back), so that a caller whose install
    fails half way still has the ones of the files that were written.
    """
    done: List[Tuple[str, str]] = []
    record = {} if record is None else record
    for rel, data in sorted(files.items()):
        target, want = agent / rel, digest(data)
        leftover = target.with_name(target.name + ".new")
        if not target.exists():
            _write(target, data)
            record[rel] = want
            done.append(("added", rel))
            continue
        have = digest(target.read_bytes())
        if have == want or force or known.get(rel) == have:
            if have != want:
                _write(target, data)
            leftover.unlink(missing_ok=True)
            record[rel] = want
            done.append(("same" if have == want else "updated", rel))
        else:
            _write(leftover, data)
            if rel in known:
                record[rel] = known[rel]
            done.append(("kept", rel))
    for rel, old in sorted(known.items()):               # written by an earlier release, not part of this one
        if rel in files or not (agent / rel).is_file():
            continue
        if digest((agent / rel).read_bytes()) == old or force:
            (agent / rel).unlink()
            done.append(("removed", rel))
        else:
            record[rel] = old
            done.append(("kept", rel))
    return done, record


def merge_block(path: Path, block: str, pattern: "re.Pattern[str]", what: str, recorded: Optional[str],
                force: bool = False) -> Tuple[str, Optional[str]]:
    """Puts ``block`` into the text file at ``path``: in place of the block that is there, else at the end.

    Returns (what happened, the digest to record): created, added, updated, same or kept. A block that is there and
    is not what the last install wrote is the person's by now (the README lets them restrict the rules, say): it is
    kept unless ``force``. Everything outside the block stays as it was, and so do the line endings of the file.
    """
    try:
        raw: Optional[bytes] = path.read_bytes()
    except FileNotFoundError:
        raw = None
    if raw is None:
        _write(path, (block + "\n").encode("utf-8"))
        return "created", digest(block.encode("utf-8"))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise InstallError(f"{path.name} is not UTF-8 text ({e.reason}): add the JINX {what} to it by hand") from e
    bom = "﻿" if text.startswith("﻿") else ""
    plain = text[len(bom):].replace("\r\n", "\n")
    found = pattern.search(plain)
    if found:
        current = found.group(0)
        if current == block:
            return "same", digest(block.encode("utf-8"))
        if not force and digest(current.encode("utf-8")) != recorded:
            return "kept", recorded
        merged, status = pattern.sub(lambda _m: block, plain, count=1), "updated"
    else:
        merged, status = (plain.rstrip("\n") + "\n\n" if plain.strip() else "") + block + "\n", "added"
    out = merged.replace("\n", "\r\n") if "\r\n" in text else merged
    _write(path, (bom + out).encode("utf-8"))
    return status, digest(block.encode("utf-8"))


def put_file(path: Path, text: str, recorded: Optional[str], force: bool = False) -> Tuple[str, Optional[str]]:
    """Writes a file that is JINX's from the first line to the last (a skill). Returns (what happened, the digest to record):
    created, updated, same or kept. A file that is there and is not what the last install wrote is the person's by now:
    it is kept unless ``force``."""
    data = text.encode("utf-8")
    want = digest(data)
    try:
        have: Optional[str] = digest(path.read_bytes())
    except FileNotFoundError:
        have = None
    if have is None:
        _write(path, data)
        return "created", want
    if have == want:
        return "same", want
    if force or have == recorded:
        _write(path, data)
        return "updated", want
    return "kept", recorded


def _rules_state(path: Path) -> str:
    """``ok``: the JINX rules are in the file, between their markers. ``legacy``: they are there without the markers (an
    install by hand, before ``jinx install`` existed). ``none``: they are not (or the file is not there, or not text)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "none"
    if _RULES.search(text):
        return "ok"
    return "legacy" if _LEGACY_RULES.search(text) else "none"


def _has_legacy_rules(path: Path) -> bool:
    """The JINX rules are in the file without their markers (see ``_rules_state``)."""
    return _rules_state(path) == "legacy"


def _has_own_text(path: Path) -> bool:
    """The file holds text of the project's own, besides the JINX rules."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(_RULES.sub("", text.replace("\r\n", "\n")).strip())


def _write_target(root: Path, name: str) -> Tuple[Optional[Path], str]:
    """(the file ``name`` of the project is written to, what to say about it). ``None``: it is not written.

    A link that leads into the project is followed (CLAUDE.md -> AGENTS.md is a usual way to keep one file of rules), so
    that writing does not replace the link by a file of its own. A link that leads out of the project is not: the file
    it names is shared with other projects, and what JINX writes does not belong in it.
    """
    path = root / name
    real = Path(os.path.realpath(path))
    if real == path:
        return path, ""
    if real != root and root not in real.parents:
        return None, f"a link to {real}, outside this project"
    return real, f"a link to {real.relative_to(root).as_posix()}"


def _block(source: Path, pattern: "re.Pattern[str]", what: str) -> str:
    """The block of ``source`` that ``pattern`` finds between its markers: the text a project is given (``what`` names it)."""
    try:
        m = pattern.search(source.read_text(encoding="utf-8").replace("\r\n", "\n"))
    except OSError as e:
        raise InstallError(f"cannot read the JINX {what} ({e})") from e
    if not m:
        raise InstallError(f"{source} has no JINX {what} block between its markers")
    return m.group(0)


def _new_name(name: str) -> str:
    """The name, under .agent/, that this release's version of the project's file ``name`` is left under when the file stays."""
    return _NEW_NAME.get(name, f"{name}.new")


# --------------------------------------------------------------------------- install
def install(root: Path, force: bool = False, shell: Optional[str] = None, out=print) -> int:
    """Installs JINX into the project at ``root`` (``jinx install``), then runs ``check``. Returns the exit code: 0 when the
    project is ready for JINX.

    ``force`` replaces what was changed since the last install; ``shell`` is the shell Claude Code runs the skill in (by
    default the one the skill of the project is written for); ``out`` prints the report.
    """
    root = root.resolve()
    if not root.is_dir():
        out(f"jinx: {root} is not a folder")
        return 1
    if not force and root in (Path.home().resolve(), Path(root.anchor)):
        out(f"jinx: {root} is not a project folder: the instruction files there (CLAUDE.md, AGENTS.md, GEMINI.md) would send "
            "every editor session below it through JINX. Run this in the project's folder, or give it as DIR "
            "(--force installs here anyway)")
        return 1
    try:
        kit = find_kit()
        rules = _block(kit.rules, _RULES, "rules")
        ignore = _block(kit.ignore, _IGNORE, ".gitignore lines")
    except InstallError as e:
        out(f"jinx: {e}")
        return 1
    agent = root / AGENT_DIR
    out(f"JINX {kit.version} -> {root}")
    files: Dict[str, str] = {}
    blocks: Dict[str, str] = {}
    progress: Dict[str, str] = {}
    own_tree, written = True, False             # a manifest is kept only where the runtime is written (not in JINX's own tree)
    try:
        known = _load_manifest(agent)
        files, blocks = dict(known["files"]), dict(known["blocks"])
        own_tree = (agent / "src" / "jinx").resolve() == kit.package.resolve()
        if own_tree:
            out("  .agent/        this is JINX's own source tree: the runtime is here already")
        else:
            busy = _active_runs(agent)
            if busy and (agent / "jinx.py").exists() and not force:
                out("jinx: a JINX run is in progress here (" + ", ".join(busy[:4]) + "): let it finish, then run this again "
                    "(or add --force to update the runtime under it)")
                return 1
            written = True
            try:
                done, files = install_runtime(agent, runtime_files(kit), known["files"], force, progress)
            except OSError:
                files = {**files, **progress}           # the files it wrote before it failed are on record all the same
                raise
            _save_manifest(agent, kit.version, files, blocks)       # at once: a failure below must not leave the runtime unrecorded
            if (agent / BASELINE).exists():             # --force: the baseline is the old release, it would undo this one
                shutil.rmtree(agent / BASELINE, ignore_errors=True)
                out(f"  .agent/        dropped {BASELINE}, the old runtime's reference for self-patches")
            tally: Dict[str, int] = {}
            for status, _rel in done:
                tally[status] = tally.get(status, 0) + 1
            out("  .agent/        " + ", ".join(f"{n} {s}" for s, n in sorted(tally.items())))
            for _status, rel in (d for d in done if d[0] == "kept"):
                out(f"                 kept {AGENT_DIR}/{rel}: it is not what `jinx install` wrote; this release's is {AGENT_DIR}/{rel}.new"
                    + ("" if force else " (`jinx install --force` replaces it)"))
        wanted = [(name, rules, _RULES, "rules") for name, _editor in INSTRUCTION_FILES]
        wanted.append((".gitignore", ignore, _IGNORE, ".gitignore lines"))
        created: List[str] = []
        for name, block, pattern, what in wanted:
            label = "JINX rules" if name != ".gitignore" else "JINX lines"
            new_name = _new_name(name)
            target, note = _write_target(root, name)
            if target is None:
                _write(agent / new_name, (block + "\n").encode("utf-8"))
                out(f"  {name:<14} is {note}: left as it is. Put the JINX {what} ({AGENT_DIR}/{new_name} has them) into "
                    "that file by hand, or replace the link by a file")
                continue
            if name != ".gitignore" and _has_legacy_rules(target):
                _write(agent / new_name, (block + "\n").encode("utf-8"))
                out(f"  {name:<14} {label} found without the markers (a copy made by hand): left as they are; "
                    f"to let `jinx install` keep them up to date, replace them with {AGENT_DIR}/{new_name}")
                continue
            status, blocks_digest = merge_block(target, block, pattern, what, blocks.get(name), force)
            if blocks_digest:
                blocks[name] = blocks_digest
            out(f"  {name:<14} {label} {status}" + (f" ({note})" if note else ""))
            if status == "created":
                created.append(name)
            if status == "kept":
                _write(agent / new_name, (block + "\n").encode("utf-8"))
                out(f"                 the block in {name} is not what `jinx install` wrote, so it stays; this release's is "
                    f"{AGENT_DIR}/{new_name}" + ("" if force else " (`jinx install --force` replaces it)"))
            elif (agent / new_name).exists():
                (agent / new_name).unlink()
        if "AGENTS.md" in created and _has_own_text(root / "CLAUDE.md"):
            out("                 OpenCode reads AGENTS.md instead of CLAUDE.md where there is one, and Kimi Code CLI reads AGENTS.md "
                "only: they see the project's own rules in CLAUDE.md only if they are in AGENTS.md too (or one of the two "
                "files is a link to the other)")
        python = (_interpreters(root) or ["python"])[0]
        shell = shell or _chosen_shell(root)
        for name, editors in SKILLS:
            new_name = _new_name(name)
            target, note = _write_target(root, name)
            text = skill_text(name, python, shell)
            if target is None:
                _write(agent / new_name, text.encode("utf-8"))
                out(f"  {name:<14} is {note}: left as it is ({AGENT_DIR}/{new_name} has the skill)")
                continue
            status, skill_digest = put_file(target, text, blocks.get(name), force)
            if skill_digest:
                blocks[name] = skill_digest
            out(f"  {'/jinx':<14} {status} in {name} ({editors})" + (f" ({note})" if note else ""))
            if status == "kept":
                _write(agent / new_name, text.encode("utf-8"))
                out(f"                 {name} is not what `jinx install` wrote, so it stays; this release's is "
                    f"{AGENT_DIR}/{new_name}" + ("" if force else " (`jinx install --force` replaces it)"))
            elif (agent / new_name).exists():
                (agent / new_name).unlink()
        if not own_tree:
            _save_manifest(agent, kit.version, files, blocks)
    except (OSError, InstallError) as e:
        if written:                                     # what was written before the failure is on record, as far as it got
            try:
                _save_manifest(agent, kit.version, files, blocks)
            except OSError:
                pass
        out(f"jinx: {e}")
        return 1
    return 0 if report(check(root, kit), out) else 1


# --------------------------------------------------------------------------- doctor
Check = Tuple[str, str, str]            # (ok | warn | FAIL, what, hint)


def _run_full(cmd: Sequence[str], cwd: Path, timeout: int = 120) -> Tuple[int, str, str]:
    """Runs a command in ``cwd``: (its exit code, what it printed, what it printed as an error). 127 where it does not start."""
    try:
        r = subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return 127, "", str(e)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _run(cmd: Sequence[str], cwd: Path, timeout: int = 120) -> Tuple[int, str]:
    """Runs a command in ``cwd``: (its exit code, what it said: the error text if there is one, else the output)."""
    code, stdout, stderr = _run_full(cmd, cwd, timeout)
    return code, (stderr or stdout)


def _interpreters(cwd: Path) -> List[str]:
    """The commands that start Python 3.10 or newer on this machine, in the order the skills prefer them."""
    probe = "import sys; sys.exit(0 if sys.version_info >= %r else 1)" % (_PYTHON_MIN,)
    return [n for n in ("python", "python3", "py") if shutil.which(n) and _run([n, "-c", probe], cwd, 30)[0] == 0]


_SKILL_COMMAND = re.compile(r"^\s*(\S+) \.agent/jinx\.py ", re.M)


def _skill_python(text: str) -> Optional[str]:
    """The command that starts Python in the JINX line of a skill (``python``, ``python3``, ``py``); None where there is none."""
    m = _SKILL_COMMAND.search(text)
    return m.group(1) if m else None


def check(root: Path, kit: Optional[Kit] = None) -> List[Check]:
    """Everything that decides whether JINX runs in ``root``, one line each."""
    out: List[Check] = []
    agent = root / AGENT_DIR
    runtime = (agent / "jinx.py").is_file() and (agent / "src" / "jinx" / "cli.py").is_file()
    out.append(("ok" if runtime else "FAIL", ".agent/ runtime",
                "" if runtime else "run `jinx install` in this folder"))
    if runtime:
        have = read_version(agent / "src" / "jinx" / "__init__.py") or "unknown"
        latest = kit.version if kit else have
        out.append(("ok" if have == latest else "warn", f"runtime version {have}",
                    "" if have == latest else f"this `jinx` is {latest}: `jinx install` updates it"))
        own = (agent / "src" / "jinx").resolve() == PKG.resolve()          # JINX's own tree has its license at the top
        licensed = (agent / LICENSE_FILES[0]).is_file() or (own and (root / LICENSE_FILES[0]).is_file())
        out.append(("ok" if licensed else "warn", "the text of the JINX license is next to the runtime",
                    "" if licensed else f"`jinx install` puts {LICENSE_FILES[0]} into {AGENT_DIR}/"))
    works = _interpreters(root)
    usable = works[0] if works else None
    out.extend(_check_editors(root, kit, usable))
    ignore_ok = False
    try:
        ignore_ok = bool(_IGNORE.search((root / ".gitignore").read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError):
        pass
    out.append(("ok" if ignore_ok else "warn", ".gitignore keeps the runtime state out of git",
                "" if ignore_ok else "`jinx install` adds the lines"))
    if usable is None:
        out.append(("FAIL", "Python %d.%d or newer" % _PYTHON_MIN, "install it, and make `python` start it"))
    else:
        out.append(("ok", f"`{usable}` runs Python {_PYTHON_MIN[0]}.{_PYTHON_MIN[1]}+", ""))
        code, why = _run([usable, "-c", "import yaml, pydantic; assert int(pydantic.VERSION.split('.')[0]) >= 2"], root, 60)
        out.append(("ok" if code == 0 else "FAIL", "pyyaml and pydantic 2 are importable",
                    "" if code == 0 else f"{usable} -m pip install \"pyyaml>=6.0\" \"pydantic>=2.0.0\"  (the first JINX run "
                                         "tries this itself; where Python is externally managed, use a virtual environment)"))
        if runtime and code == 0:
            code, why = _run([usable, "-c", "import sys; sys.path.insert(0, '.agent/src'); "
                                            "import jinx.cli, jinx.runner, jinx.sessions"], root, 60)
            out.append(("ok" if code == 0 else "FAIL", "the runtime loads",
                        "" if code == 0 else (why.splitlines() or ["see the error above"])[-1][:200]))
            # The call the /jinx skill of Claude Code makes (its words are --peek: it looks at the sessions and changes nothing).
            # Made through --via it ends with 0 even when it failed (the code is printed as `[JINX_EXIT] <code>`): look for that too.
            code, stdout, stderr = _run_full([usable, f"{AGENT_DIR}/jinx.py", "--via", "/jinx", "--args", "--peek"], root, 60)
            reached = code == 0 and not re.search(r"^\[JINX_EXIT\]", stdout, re.M)
            said = [line for line in (stderr or stdout).splitlines() if line.strip()]
            out.append(("ok" if reached else "FAIL", "/jinx reaches the runtime",
                        "" if reached else (said or ["it printed nothing"])[-1][:200]))
    code, top = _run(["git", "rev-parse", "--show-toplevel"], root, 30)
    if code != 0:
        out.append(("warn", "git repository", "not one (or no git): JINX works, but writes no round files (.mbox/.yaml) "
                                              "and has nothing to export"))
    elif Path(top).resolve() != root.resolve():
        out.append(("warn", "git repository", f"this folder is inside the repository at {top}: round files need the top of "
                                              "the repository as the project root"))
    else:
        out.append(("ok", "git repository", ""))
    editors = [e for e in _EDITORS if shutil.which(e)]
    out.append(("ok" if editors else "warn", "an editor is on PATH",
                ", ".join(editors) if editors else "JINX needs an editor to drive it: Claude Code (`claude`), Codex CLI "
                                                   "(`codex`), OpenCode (`opencode`), Gemini CLI (`gemini`), Grok Build "
                                                   "(`grok`) or Kimi Code CLI (`kimi`)"))
    return out


def _check_editors(root: Path, kit: Optional[Kit], usable: Optional[str]) -> List[Check]:
    """Per editor: its instruction file has to hold the JINX rules, and the skill that is `/jinx` has to be there, or the
    editor does not use JINX (rules without the skill are worse than neither: the editor is sent to a command that is not there)."""
    current: Optional[str] = None
    if kit:
        try:
            current = _block(kit.rules, _RULES, "rules")
        except InstallError:
            pass
    states = [(name, editor, _rules_state(root / name)) for name, editor in INSTRUCTION_FILES]
    told = any(state != "none" for _name, _editor, state in states)      # at least one editor is told to use JINX
    out: List[Check] = []
    for name, editor, state in states:
        target, note = _write_target(root, name)
        shared = target is None                     # a link out of the project: `jinx install` does not write there
        if state == "ok":
            block = _RULES.search((root / name).read_text(encoding="utf-8").replace("\r\n", "\n"))
            if current is not None and block is not None and block.group(0) != current:
                fix = (f"{name} is {note}, which `jinx install` leaves alone: copy this release's block from "
                       f"{AGENT_DIR}/{_new_name(name)} into that file" if shared else
                       f"`jinx install` updates them (a block that was changed by hand is kept, and this release's is "
                       f"written to {AGENT_DIR}/{_new_name(name)})")
                out.append(("warn", f"{name} has JINX rules that are not this release's", f"{editor}: {fix}"))
            else:
                out.append(("ok", f"{name} has the JINX rules", editor))
        elif state == "legacy":
            out.append(("warn", f"{name} has the JINX rules, without the markers",
                        "they work, but `jinx install` cannot keep them up to date: replace them with the block in "
                        f"{AGENT_DIR}/{_new_name(name)} (written by `jinx install`)"))
        else:
            fix = (f"{name} is {note}, which `jinx install` leaves alone: put the JINX rules into that file by hand "
                   f"(`jinx install` leaves them in {AGENT_DIR}/{_new_name(name)}), or replace the link by a file" if shared else
                   "`jinx install` adds them")
            out.append(("warn" if told else "FAIL", f"{name} has the JINX rules",
                        f"without them tasks are not routed through JINX in {editor}: {fix}"))
    wants_claude = states[0][2] != "none"
    wants_agents = any(state != "none" for _name, _editor, state in states[1:])
    for name, editors in SKILLS:
        needed = wants_claude if name == CLAUDE_SKILL else wants_agents
        path = root / name
        try:
            text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        except (OSError, UnicodeDecodeError):
            target, note = _write_target(root, name)
            fix = (f"{name} is {note}, which `jinx install` leaves alone" if target is None else "`jinx install` writes it")
            out.append(("FAIL" if needed else "warn", f"/jinx is in {name}",
                        f"{editors} have no such command without it, and their rules send them to it: {fix}"
                        if needed else f"{editors} have no /jinx without it: {fix}"))
            continue
        python = _skill_python(text)
        shell = _skill_shell(text)
        if python is None:
            out.append(("warn", f"/jinx in {name} runs no JINX command",
                        f"it was changed by hand: `jinx install` keeps it, and writes this release's to {AGENT_DIR}/{_new_name(name)}"))
        elif text != skill_text(name, python, shell):
            out.append(("warn", f"/jinx in {name} is not this release's",
                        f"{editors}: `jinx install` updates it (a file that was changed by hand is kept, and this release's is "
                        f"written to {AGENT_DIR}/{_new_name(name)})"))
        elif shutil.which(python) is None:
            out.append(("FAIL", f"/jinx in {name} runs `{python}`, which this machine does not have",
                        f"`jinx install` writes the command that works here ({usable or 'no Python 3.10+ found'})"))
        else:
            out.append(("ok", f"/jinx is in {name}", editors + (" (runs in PowerShell)" if name == CLAUDE_SKILL and shell == "powershell"
                                                                 else "")))
    return out


def report(checks: Sequence[Check], out=print) -> bool:
    """Prints the checks, then one line for the person. True when nothing failed."""
    for level, what, hint in checks:
        out(f"  {level:<5} {what}" + (f": {hint}" if hint else ""))
    failed = [c for c in checks if c[0] == "FAIL"]
    if failed:
        out(f"{len(failed)} problem(s) above. Fix them and run `jinx doctor` again.")
        return False
    out("JINX is ready. Open this folder in your editor (Claude Code, Codex CLI, OpenCode, Gemini CLI, Grok Build or Kimi Code "
        "CLI) and give it a task: every message goes through JINX.")
    return True


# --------------------------------------------------------------------------- command line
def main(argv: Optional[Sequence[str]] = None) -> int:
    """The ``jinx`` command (``install``, ``doctor``). Returns the exit code."""
    for stream in (sys.stdout, sys.stderr):                 # a pipe is cp1251 or cp1252 on Windows: a name must not end the run
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0].lower() in ("install", "init", "doctor"):       # `jinx Install` is `jinx install`; `init` is its old name
        args_in[0] = {"init": "install"}.get(args_in[0].lower(), args_in[0].lower())
    kit_version = read_version(PKG / "__init__.py") or "unknown"
    ap = argparse.ArgumentParser(
        prog="jinx", description="Install JINX into a project, and check that it can run there.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="two commands install it:\n"
               "  pip install git+https://github.com/machinegpt/agent\n"
               "  jinx install                 (in the project folder)")
    ap.add_argument("--version", action="version", version=f"jinx {kit_version}")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("install", help="put JINX into a project (the current folder unless DIR is given); run it again to update")
    p.add_argument("dir", nargs="?", default=".")
    p.add_argument("--force", action="store_true", help="replace runtime files even where they changed since the install")
    p.add_argument("--claude-shell", choices=("bash", "powershell"), default=None,
                   help="the shell Claude Code runs /jinx in (default: the one the project's skill is written for, else "
                        "powershell on Windows and bash elsewhere)")
    p = sub.add_parser("doctor", help="check that a project can run JINX")
    p.add_argument("dir", nargs="?", default=".")
    args = ap.parse_args(args_in)
    if args.cmd == "install":
        return install(Path(args.dir), force=args.force, shell=args.claude_shell)
    if args.cmd == "doctor":
        root = Path(args.dir).resolve()
        try:
            kit: Optional[Kit] = find_kit()
        except InstallError:
            kit = None
        return 0 if report(check(root, kit)) else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
