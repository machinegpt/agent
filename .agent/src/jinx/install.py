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
"""Installs JINX into a project: ``jinx init`` and ``jinx doctor``.

    pip install git+https://github.com/machinegpt/agent      # 1. the `jinx` command
    jinx init                                                 # 2. in the project folder: set JINX up there

``init`` puts the runtime into ``.agent/`` (``jinx.py`` and ``src/jinx/``), the JINX rules into ``CLAUDE.md`` and the
lines that keep the runtime state out of git into ``.gitignore``, then runs ``doctor``. Run it again after
``pip install -U`` and it brings the runtime up to date: a file that was changed since it was installed (by a person,
or by JINX improving its own source) is left alone, and the new version is written beside it as ``<name>.new``.

CLAUDE.md and .gitignore are not owned by JINX: only the text between its markers is ever written, and a project's own
text around it is kept as it is, line endings included. Nothing here needs more than the standard library.
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
# The text a project gets in CLAUDE.md / .gitignore is the text between these markers in the repository's own files.
_RULES = re.compile(r"<!--\s*JINX:BEGIN\b.*?-->.*?<!--\s*JINX:END\s*-->", re.S)
_IGNORE = re.compile(r"# >>> JINX >>>.*?# <<< JINX <<<", re.S)
_LEGACY_RULES = re.compile(r"route absolutely every single user message")     # the rules of a copy made without `jinx init`
_VERSION = re.compile(r"""__version__\s*=\s*["']([^"']+)["']""")
_PYTHON_MIN = (3, 10)


class InstallError(Exception):
    """Something the person has to fix; the message says what."""


class Kit:
    """What one JINX release hands a project: the runtime, the rules for CLAUDE.md and the lines for .gitignore."""

    def __init__(self, entry: Path, rules: Path, ignore: Path, package: Path):
        self.entry, self.rules, self.ignore, self.package = entry, rules, ignore, package

    @property
    def version(self) -> str:
        return read_version(self.package / "__init__.py") or "unknown"


def find_kit() -> Kit:
    """The kit of this copy of JINX: packed next to the package by the build (``pip install``), or the checkout itself."""
    packed = PKG / "_kit"
    if (packed / "jinx.py").is_file():
        return Kit(packed / "jinx.py", packed / "CLAUDE.md", packed / "gitignore", PKG)
    agent = PKG.parent.parent                       # a checkout: .agent/src/jinx
    root = agent.parent
    kit = Kit(agent / "jinx.py", root / "CLAUDE.md", root / ".gitignore", PKG)
    missing = [p.name for p in (kit.entry, kit.rules, kit.ignore) if not p.is_file()]
    if missing:
        raise InstallError("this copy of JINX has no " + ", ".join(missing) + " to install from: "
                           "install the whole repository (pip install git+https://github.com/machinegpt/agent)")
    return kit


def read_version(init: Path) -> Optional[str]:
    try:
        m = _VERSION.search(init.read_text(encoding="utf-8"))
    except OSError:
        return None
    return m.group(1) if m else None


# --------------------------------------------------------------------------- files
def digest(data: bytes) -> str:
    """Line endings do not count: a checkout that converts them (``core.autocrlf``) has not changed a file."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".jinx-tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def runtime_files(kit: Kit) -> Dict[str, bytes]:
    """{path under .agent/: content} of the runtime of this release."""
    files = {"jinx.py": kit.entry.read_bytes()}
    for p in sorted(kit.package.glob("*.py")):
        files[f"src/jinx/{p.name}"] = p.read_bytes()
    return files


def _load_manifest(agent: Path) -> Dict[str, Dict[str, str]]:
    """What the last install wrote: {"files": {path under .agent/: digest}, "blocks": {file of the project: digest}}."""
    out: Dict[str, Dict[str, str]] = {"files": {}, "blocks": {}}
    try:
        data = json.loads((agent / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for key in out:
        part = data.get(key) if isinstance(data, dict) else None
        if isinstance(part, dict):
            out[key] = {str(k): str(v) for k, v in part.items()}
    return out


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


def install_runtime(agent: Path, files: Dict[str, bytes], known: Dict[str, str],
                    force: bool = False) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Writes the runtime into ``agent``. Returns (what happened, path) per file (added, updated, same, kept, removed)
    and the digests to record.

    A file that differs from the release is replaced only when it is what the last install wrote (nobody changed
    it since) or with ``force``; otherwise it stays, and the release's version is written beside it as ``.new``.
    """
    done: List[Tuple[str, str]] = []
    record: Dict[str, str] = {}
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
    bom = "\ufeff" if text.startswith("\ufeff") else ""
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


def _has_legacy_rules(path: Path) -> bool:
    """The JINX rules are in the file, but not between the markers: an install by hand, before ``jinx init`` existed."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(_LEGACY_RULES.search(text)) and not _RULES.search(text)


def _block(source: Path, pattern: "re.Pattern[str]", what: str) -> str:
    try:
        m = pattern.search(source.read_text(encoding="utf-8").replace("\r\n", "\n"))
    except OSError as e:
        raise InstallError(f"cannot read the JINX {what} ({e})") from e
    if not m:
        raise InstallError(f"{source} has no JINX {what} block between its markers")
    return m.group(0)


# --------------------------------------------------------------------------- init
def init(root: Path, force: bool = False, out=print) -> int:
    root = root.resolve()
    if not root.is_dir():
        out(f"jinx: {root} is not a folder")
        return 1
    if not force and root in (Path.home().resolve(), Path(root.anchor)):
        out(f"jinx: {root} is not a project folder: CLAUDE.md there would send every Claude Code session below it through "
            "JINX. Run this in the project's folder, or give it as DIR (--force installs here anyway)")
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
    try:
        known = _load_manifest(agent)
        files = dict(known["files"])
        own_tree = (agent / "src" / "jinx").resolve() == kit.package.resolve()
        if own_tree:
            out("  .agent/        this is JINX's own source tree: the runtime is here already")
        else:
            busy = _active_runs(agent)
            if busy and (agent / "jinx.py").exists() and not force:
                out("jinx: a JINX run is in progress here (" + ", ".join(busy[:4]) + "): let it finish, then run this again "
                    "(or add --force to update the runtime under it)")
                return 1
            done, files = install_runtime(agent, runtime_files(kit), known["files"], force)
            if (agent / BASELINE).exists():             # --force: the baseline is the old release, it would undo this one
                shutil.rmtree(agent / BASELINE, ignore_errors=True)
                out(f"  .agent/        dropped {BASELINE}, the old runtime's reference for self-patches")
            tally: Dict[str, int] = {}
            for status, _rel in done:
                tally[status] = tally.get(status, 0) + 1
            out("  .agent/        " + ", ".join(f"{n} {s}" for s, n in sorted(tally.items())))
            for _status, rel in (d for d in done if d[0] == "kept"):
                out(f"                 kept {AGENT_DIR}/{rel}: it is not what `jinx init` wrote; this release's is {AGENT_DIR}/{rel}.new"
                    + ("" if force else " (`jinx init --force` replaces it)"))
        blocks = dict(known["blocks"])
        for name, block, pattern, what, new_name in (("CLAUDE.md", rules, _RULES, "rules", "CLAUDE.md.new"),
                                                     (".gitignore", ignore, _IGNORE, ".gitignore lines", "gitignore.new")):
            label = "JINX rules" if name == "CLAUDE.md" else "JINX lines"
            if name == "CLAUDE.md" and _has_legacy_rules(root / name):
                _write(agent / new_name, (block + "\n").encode("utf-8"))
                out(f"  {name:<14} {label} found without the markers (a copy made by hand): left as they are; "
                    f"to let `jinx init` keep them up to date, replace them with {AGENT_DIR}/{new_name}")
                continue
            status, blocks_digest = merge_block(root / name, block, pattern, what, blocks.get(name), force)
            if blocks_digest:
                blocks[name] = blocks_digest
            out(f"  {name:<14} {label} {status}")
            if status == "kept":
                _write(agent / new_name, (block + "\n").encode("utf-8"))
                out(f"                 the block in {name} is not what `jinx init` wrote, so it stays; this release's is "
                    f"{AGENT_DIR}/{new_name}" + ("" if force else " (`jinx init --force` replaces it)"))
            elif (agent / new_name).exists():
                (agent / new_name).unlink()
        if not own_tree:
            _write(agent / MANIFEST, (json.dumps({"jinx": kit.version, "files": files, "blocks": blocks}, indent=2,
                                                 sort_keys=True) + "\n").encode("utf-8"))
    except (OSError, InstallError) as e:
        out(f"jinx: {e}")
        return 1
    return 0 if report(check(root, kit), out) else 1


# --------------------------------------------------------------------------- doctor
Check = Tuple[str, str, str]            # (ok | warn | FAIL, what, hint)


def _run(cmd: Sequence[str], cwd: Path, timeout: int = 120) -> Tuple[int, str]:
    try:
        r = subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return 127, str(e)
    return r.returncode, (r.stderr.strip() or r.stdout.strip())


def _interpreter(cwd: Path) -> Tuple[Optional[str], Optional[str]]:
    """(the command the rules say, a command that works) for Python 3.10 or newer on this machine."""
    probe = "import sys; sys.exit(0 if sys.version_info >= %r else 1)" % (_PYTHON_MIN,)
    works = [n for n in ("python", "python3", "py") if shutil.which(n) and _run([n, "-c", probe], cwd, 30)[0] == 0]
    return ("python" if "python" in works else None), (works[0] if works else None)


def check(root: Path, kit: Optional[Kit] = None) -> List[Check]:
    """Everything that decides whether JINX runs in ``root``, one line each."""
    out: List[Check] = []
    agent = root / AGENT_DIR
    runtime = (agent / "jinx.py").is_file() and (agent / "src" / "jinx" / "cli.py").is_file()
    out.append(("ok" if runtime else "FAIL", ".agent/ runtime",
                "" if runtime else "run `jinx init` in this folder"))
    if runtime:
        have = read_version(agent / "src" / "jinx" / "__init__.py") or "unknown"
        latest = kit.version if kit else have
        out.append(("ok" if have == latest else "warn", f"runtime version {have}",
                    "" if have == latest else f"this `jinx` is {latest}: `jinx init` updates it"))
    rules_ok = False
    try:
        rules_ok = bool(_RULES.search((root / "CLAUDE.md").read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError):
        pass
    if rules_ok:
        out.append(("ok", "CLAUDE.md has the JINX rules", ""))
    elif _has_legacy_rules(root / "CLAUDE.md"):
        out.append(("warn", "CLAUDE.md has the JINX rules, without the markers",
                    "they work, but `jinx init` cannot keep them up to date: replace them with the block in "
                    f"{AGENT_DIR}/CLAUDE.md.new (written by `jinx init`)"))
    else:
        out.append(("FAIL", "CLAUDE.md has the JINX rules",
                    "without them the editor does not route tasks through JINX: `jinx init`"))
    ignore_ok = False
    try:
        ignore_ok = bool(_IGNORE.search((root / ".gitignore").read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError):
        pass
    out.append(("ok" if ignore_ok else "warn", ".gitignore keeps the runtime state out of git",
                "" if ignore_ok else "`jinx init` adds the lines"))
    said, usable = _interpreter(root)
    if usable is None:
        out.append(("FAIL", "Python %d.%d or newer" % _PYTHON_MIN, "install it, and make `python` start it"))
    else:
        out.append(("ok" if said else "warn", f"`{usable}` runs Python {_PYTHON_MIN[0]}.{_PYTHON_MIN[1]}+",
                    "" if said else "the rules and JINX's messages say `python .agent/jinx.py`: tell the editor to say "
                                    f"`{usable}`, or make `python` available (python-is-python3, an alias)"))
        code, why = _run([usable, "-c", "import yaml, pydantic; assert int(pydantic.VERSION.split('.')[0]) >= 2"], root, 60)
        out.append(("ok" if code == 0 else "FAIL", "pyyaml and pydantic 2 are importable",
                    "" if code == 0 else f"{usable} -m pip install \"pyyaml>=6.0\" \"pydantic>=2.0.0\"  (the first JINX run "
                                         "tries this itself; where Python is externally managed, use a virtual environment)"))
        if runtime and code == 0:
            code, why = _run([usable, "-c", "import sys; sys.path.insert(0, '.agent/src'); "
                                            "import jinx.cli, jinx.runner, jinx.sessions"], root, 60)
            out.append(("ok" if code == 0 else "FAIL", "the runtime loads",
                        "" if code == 0 else (why.splitlines() or ["see the error above"])[-1][:200]))
            code, why = _run([usable, f"{AGENT_DIR}/jinx.py", "--peek"], root, 60)
            out.append(("ok" if code == 0 else "FAIL", "`python .agent/jinx.py` starts",
                        "" if code == 0 else (why.splitlines() or [""])[-1][:200]))
    code, top = _run(["git", "rev-parse", "--show-toplevel"], root, 30)
    if code != 0:
        out.append(("warn", "git repository", "not one (or no git): JINX works, but writes no round files (.mbox/.yaml) "
                                              "and has nothing to export"))
    elif Path(top).resolve() != root.resolve():
        out.append(("warn", "git repository", f"this folder is inside the repository at {top}: round files need the top of "
                                              "the repository as the project root"))
    else:
        out.append(("ok", "git repository", ""))
    out.append(("ok" if shutil.which("claude") else "warn", "Claude Code (`claude`) is on PATH",
                "" if shutil.which("claude") else "JINX needs an editor to drive it: https://claude.com/product/claude-code"))
    return out


def report(checks: Sequence[Check], out=print) -> bool:
    """Prints the checks, then one line for the person. True when nothing failed."""
    for level, what, hint in checks:
        out(f"  {level:<5} {what}" + (f": {hint}" if hint else ""))
    failed = [c for c in checks if c[0] == "FAIL"]
    if failed:
        out(f"{len(failed)} problem(s) above. Fix them and run `jinx doctor` again.")
        return False
    out("JINX is ready. Open this folder in Claude Code (`claude`) and give it a task: every message goes through JINX.")
    return True


# --------------------------------------------------------------------------- command line
def main(argv: Optional[Sequence[str]] = None) -> int:
    kit_version = read_version(PKG / "__init__.py") or "unknown"
    ap = argparse.ArgumentParser(
        prog="jinx", description="Install JINX into a project, and check that it can run there.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="two commands install it:\n"
               "  pip install git+https://github.com/machinegpt/agent\n"
               "  jinx init                    (in the project folder)")
    ap.add_argument("--version", action="version", version=f"jinx {kit_version}")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("init", help="put JINX into a project (the current folder unless DIR is given); run it again to update")
    p.add_argument("dir", nargs="?", default=".")
    p.add_argument("--force", action="store_true", help="replace runtime files even where they changed since the install")
    p = sub.add_parser("doctor", help="check that a project can run JINX")
    p.add_argument("dir", nargs="?", default=".")
    args = ap.parse_args(argv)
    if args.cmd == "init":
        return init(Path(args.dir), force=args.force)
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
