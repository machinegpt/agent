#!/usr/bin/env python3
"""
Direct entry point for JINX.
Designed to be executed when the repository is dropped into a project as a `.agent` folder
(`pip install git+https://github.com/machinegpt/agent`, then `jinx install` in the project, does that for you).

An editor calls it as its own command, `/jinx` (the skill that `jinx install` sets up for it).
"""
import re
import sys
from pathlib import Path

# Insert the local src/ directory into sys.path so 'import jinx' resolves correctly
# without needing a global pip installation.
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))


# An editor that has JINX as a command of its own (Claude Code: the `/jinx` skill) does not build a shell command line
# out of what the person wrote. The skill hands the words over in one piece:
#
#     jinx.py --via /jinx --stdin       the words are on stdin (a here-document of the skill)
#     jinx.py --via /jinx --args ...    the words are everything after --args (for a shell without here-documents)
#
# so a message with quotes, `$`, backticks or any language in it arrives exactly as it was written. The words are
# either options of JINX (`--session s3fa1`, `--peek s7b2e --diff`, or nothing at all: go on) or a message to start a run with.
# `--via NAME` is the name the editor calls JINX by: it is what JINX tells the editor to run next (JINX_CMD, see
# mail.jinx_command). Without --via none of this applies, and a call by hand is parsed as it always was.
#
# What each option takes after it, as cli.py declares them: (how many values, what a value looks like). The "how many" is
# "0", "1", "?" (at most one) or "*". This file cannot import cli.py (the self-patch preflight below has to run before
# anything of the package does), so the id of a session is written out here as it is in mail._SID_RE.
_SESSION_ID = re.compile(r"[a-z][a-z0-9]{2,15}")
_OPTIONS = {
    "--session": ("1", _SESSION_ID),
    "--join": ("1", _SESSION_ID),
    "--parent": ("1", _SESSION_ID),
    "--peek": ("?", _SESSION_ID),
    "--diff": ("0", None),
    "--full": ("0", None),
    "--export": ("*", _SESSION_ID),
    "--close": ("*", _SESSION_ID),
    "--force": ("0", None),
    "--out": ("1", re.compile(r"[^-].*", re.S)),
    "--min": ("1", re.compile(r"[0-9]+")),
    "--ipc": ("1", re.compile(r"file|rpc")),
}
_TASK_AFTER = ("--join", "--parent")    # `--join <id> ["task"]`, `--parent <id> "task"`: the calls of an editor with a message behind their options
_WORD = re.compile(r"\"[^\"]*\"|'[^']*'|\S+")


def _unquote(word):
    """``"a b"`` or ``'a b'`` as ``a b``; anything else as it is."""
    if len(word) >= 2 and word[0] in "\"'" and word[-1] == word[0] and word.count(word[0]) == 2:
        return word[1:-1]
    return word


def _message(text):
    """The words as one message. A message that begins with ``-`` goes behind ``--``, so that it is never read as an option."""
    return ["--", text] if text.startswith("-") else [text]


def _split_words(text):
    """The words of an editor's call as a command line's arguments: options as they are, anything else one message.

    The words are options only when they are nothing but options and the values those take, and, after ``--join <id>``,
    the task of the new session, which stays as it was written (blanks and line breaks included). Anything else is a
    message, even when it begins like an option: ``--peek is broken`` is a sentence of the person's and must start a
    run, not look at the sessions.
    """
    text = text.strip()
    if not text:
        return []
    words = [(m.group(), m.start()) for m in _WORD.finditer(text)]
    args, i, task_may_follow = [], 0, False
    while i < len(words):
        name, eq, value = words[i][0].partition("=")
        spec = _OPTIONS.get(name)
        if spec is None:
            break                                       # not an option: from here on it is the message
        count, shape = spec
        i += 1
        args.append(name)
        if eq:                                          # --session=s1
            if count == "0" or not shape.fullmatch(_unquote(value)):
                return _message(text)
            args.append(_unquote(value))
        else:
            taken = 0
            while (count != "0" and i < len(words) and (count == "*" or not taken)
                   and shape.fullmatch(_unquote(words[i][0]))):
                args.append(_unquote(words[i][0]))
                i += 1
                taken += 1
            if count == "1" and not taken:              # --session, and no id behind it
                return _message(text)
        task_may_follow = task_may_follow or name in _TASK_AFTER
    if i < len(words):
        if not task_may_follow:                         # words that no option takes: it was a message after all
            return _message(text)
        args.append(_unquote(text[words[i][1]:]))
    return args


def _expand_call():
    """Rewrites ``sys.argv`` for a call made through the editor's command. True when it was one."""
    args = sys.argv[1:]
    if args[:1] != ["--via"] or len(args) < 2:
        return False
    import os
    os.environ["JINX_CMD"] = args[1]
    rest = args[2:]
    if rest[:1] == ["--stdin"]:
        raw = b"" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.buffer.read()
        rest = _split_words(raw.decode("utf-8", "replace").lstrip("﻿"))
    elif rest[:1] == ["--args"]:
        rest = _split_words(" ".join(rest[1:]))
    sys.argv[1:] = rest
    return True


_VIA = _expand_call()


def _code(code):
    """The exit code to leave with.

    Through the editor's command a call is a success only when it exits 0, and one that does not is handed to the model
    as a command that failed, not as what JINX printed. So there the code is printed on a line of its own and the call
    ends well: that is how the model learns that a run ended without success (exit code 2) or that a launch failed (1).
    """
    if isinstance(code, str):       # sys.exit("message")
        print(code, file=sys.stderr)
        code = 1
    if _VIA and code not in (None, 0):
        print("[JINX_EXIT] %s" % code, flush=True)
        return 0
    return code


def _load_selfpatch():
    """Loads ``selfpatch.py`` from its file, without importing the package first.

    The module name must be package-qualified. ``selfpatch.py`` opens with
    ``from . import prompts``, and loading it under a bare name leaves
    ``__package__`` empty, so the import machinery raises "attempted relative
    import with no known parent package". The caller treats a raised loader as
    "nothing to repair", so a bare name silently disarmed the outermost brake
    on every single run while the framework stayed free to patch itself.

    Loading it still requires ``prompts.py`` to be importable, which is why
    :func:`_restore_baseline_without_selfpatch` exists as a path that does not.

    Returns the loaded module, or None when there is nothing to load.
    """
    import importlib.util

    module_path = src_path / "jinx" / "selfpatch.py"
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("jinx.selfpatch", module_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec so the relative import inside the module resolves
    # against the real package rather than re-executing the file twice.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # The normal import machinery removes a module whose execution failed.
        # Manual spec loading does not, so a half-built selfpatch would stay
        # cached: BASELINE_DIR may be set while later definitions are missing,
        # and the next import in this process would get that husk instead of
        # re-executing a file the fallback path has since repaired. Leaving
        # sys.modules clean is what lets recovery take effect at all.
        sys.modules.pop(spec.name, None)
        raise
    return mod


def _restore_baseline_without_selfpatch():
    """Last-resort restore that imports nothing from the tree it repairs.

    ``selfpatch.py`` opens with ``from . import prompts``, so loading it needs
    ``prompts.py`` to be importable. A self-edit that put a syntax error into
    ``prompts.py`` would therefore disarm the very brake meant to repair it --
    the module meant to save the framework could not be loaded *because of* the
    damage. This path closes that gap: it reads the baseline with the standard
    library only and copies files back, so the recovery does not depend on any
    module the model is allowed to edit.

    Returns the names restored, or None when there is no baseline to restore.
    """
    import shutil

    # src_path is .agent/src, so the baseline is its sibling and the sources
    # live one level below it. Mirrors selfpatch.SRC_DIR / BASELINE_DIR.
    baseline = src_path.parent / ".selfpatch_baseline"
    src_dir = src_path / "jinx"
    repo_root = src_path.parent.parent
    if not baseline.is_dir():
        return None

    restored = []
    for saved in sorted(baseline.rglob("*")):
        if not saved.is_file():
            continue
        rel = saved.relative_to(baseline)
        if rel.parts[0] == "repo":
            target = repo_root / rel.relative_to("repo")
        else:
            target = src_dir / rel
        try:
            wanted = saved.read_text(encoding="utf-8")
            if target.is_file() and target.read_text(encoding="utf-8") == wanted:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(saved, target)
            restored.append(str(rel).replace("\\", "/"))
        except OSError as e:
            print("[JINX SELF-PATCH] Could not restore %s: %s" % (rel, e),
                  file=sys.stderr)
    return restored


def self_patch_preflight():
    """Revert a self-edit that broke the framework BEFORE importing it.

    The verify-and-revert gate lives inside the runner, but a self-edit that
    introduces a syntax error into an imported module (tools.py, state.py, ...)
    means the next process cannot start at all. A gate that only runs after the
    import could therefore never catch the exact failure it exists to prevent --
    it would be disarmed by the first self-patch bad enough to matter.

    ``selfpatch.py`` is loaded here directly from its file, before and
    independently of the rest of the code it protects. This is the outermost
    layer of the brake, and it is the only one that still runs when the
    framework itself no longer imports.

    The baseline only exists once a run has started, so this is a no-op in the
    normal case and costs one stat() per source file.
    """
    import os

    if os.environ.get("JINX_SELF_PATCH", "1") in ("0", "false", "False"):
        return
    try:
        mod = _load_selfpatch()
    except Exception as e:
        # The brake itself will not load -- most likely because a self-edit
        # broke a module it imports. Fall back to the stdlib-only restore
        # instead of giving up, so the damage can still be rolled back.
        print("[JINX SELF-PATCH] Preflight degraded (%s); using fallback restore"
              % e, file=sys.stderr)
        try:
            restored = _restore_baseline_without_selfpatch()
        except Exception as inner:
            print("[JINX SELF-PATCH] Fallback restore unavailable: %s" % inner,
                  file=sys.stderr)
            return
        if restored:
            _record_feedback(
                "[JINX SELF-PATCH REVERTED] JINX's own source could not be "
                "loaded, so its safety checks could not run. The last verified "
                "state was restored by the fallback path: %s."
                % ", ".join(restored)
            )
            print("[JINX SELF-PATCH] Fallback restored: %s" % ", ".join(restored),
                  file=sys.stderr)
        return
    if mod is None:
        return

    try:
        if not mod.BASELINE_DIR.exists():
            return
        if not mod.baseline_changed():
            # Nothing was touched, so the baseline is still the right reference
            # for the rest of the run. Clearing it here would disarm the gate
            # before the first tool call ever happened.
            return
        # Several JINX processes can start at once (one per session and transition), and
        # only one of them may verify and restore: a baseline that two processes rewrite
        # together can be left half written. The lock goes away with its holder.
        with mod.gate_lock() as held:
            if not held:
                print("[JINX SELF-PATCH] Another process is running the gate; skipping the preflight",
                      file=sys.stderr)
                return
            # Looked at again under the lock: the process that held it may just have
            # verified (and adopted) or reverted exactly this change.
            changed = mod.baseline_changed()
            if not changed:
                return
            # Same protection re-check the runner does: `file_write` is not the only
            # way into the source tree, and a brake weakened via `bash_exec` would
            # otherwise be adopted here as the new baseline.
            violations = mod.protection_violations()
            if violations:
                restored = mod.restore_baseline()
                detail = "; ".join(
                    "%s: %s" % (name, ", ".join(reasons))
                    for name, reasons in sorted(violations.items())
                )
                _record_feedback(
                    "[JINX SELF-PATCH REFUSED] The previous round changed protected "
                    "brake logic (%s). It was rolled back automatically and NOT "
                    "verified: these functions are what stop a self-patch from "
                    "removing its own safety checks, so no test result can justify "
                    "changing them. Improve something else, or ask a human." % detail
                )
                print(
                    "[JINX SELF-PATCH] Refused and rolled back: %s"
                    % ", ".join(restored or sorted(violations)),
                    file=sys.stderr,
                )
                return
            # The editor CLI owns functional testing. The self-patch gate retains
            # only fast checks (compile, then import), so an edit that cannot be
            # loaded never becomes the next baseline.
            result = mod.verify()
            if result["ok"]:
                print("[JINX SELF-PATCH] Check passed (%s): %s" % (result["summary"], ", ".join(changed)),
                      file=sys.stderr)
                # The edit is structurally valid, so adopt it as the next
                # reference instead of checking the same diff on every process.
                mod.capture_baseline()
                return
            restored = mod.restore_baseline()
    except Exception as e:
        print("[JINX SELF-PATCH] Preflight failed: %s" % e, file=sys.stderr)
        return

    message = (
        "[JINX SELF-PATCH REVERTED] The previous round edited JINX's own source "
        "(%s) and failed the compile/import check, so it could not be accepted. The change was "
        "rolled back automatically and the run continues (%s).\n%s"
        % (
            ", ".join(changed),
            result["summary"],
            "\n".join("[%s] %s" % (c["name"], c["tail"]) for c in result["checks"]),
        )
    )
    print(message, file=sys.stderr)
    _record_feedback(message)


# A preflight verdict waits in this folder (next to the run state of the run it is for), one small
# file each, until that run reads and removes it (runner._take_notices). At most MAX_NOTICES wait at
# once, so a run that never comes back for them cannot collect them for ever.
NOTICES_DIR = "notices"
MAX_NOTICES = 8


def _record_feedback(message):
    """Leaves a preflight verdict for the next prompt of every run that is in progress.

    The verdict goes into a file of its own and never into a run state: the process that notices
    the damage does not own the other sessions' run states, and rewriting a file whose owner is
    about to rewrite it too silently drops one of the two writes. Every verdict gets a name of its
    own, so two preflights cannot overwrite each other either.

    pyyaml is an external dependency, not part of the package, so this still works when the source
    tree itself is too broken to import JINX.
    """
    try:
        import os
        import time
        import yaml
        agent_dir = Path(__file__).resolve().parent
        # The single shared run, and every JINX session's: the process that notices the damage is
        # not necessarily the one whose round caused it, and the message is true for all of them
        # (the source they all run was rolled back).
        runs = [agent_dir / "jinx_run_state.yaml"]
        ipc = Path(os.environ.get("JINX_IPC_DIR") or (agent_dir / "ipc"))
        try:
            runs.extend(sorted((ipc / "sessions").glob("*/jinx_run_state.yaml")))
        except OSError:
            pass
        body = yaml.safe_dump({"message": message}, allow_unicode=True, sort_keys=False)
        for run_state_path in runs:
            if not run_state_path.is_file():
                continue
            folder = run_state_path.parent / NOTICES_DIR
            folder.mkdir(exist_ok=True)
            name = "%019d-%d.yaml" % (time.time_ns(), os.getpid())
            tmp = folder / (name + ".tmp")
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, folder / name)
            for old in sorted(folder.glob("*.yaml"))[:-MAX_NOTICES]:
                old.unlink(missing_ok=True)
    except Exception as e:
        print("[JINX SELF-PATCH] Could not record feedback: %s" % e, file=sys.stderr)


def bootstrap_dependencies():
    """Installs pydantic and pyyaml with pip when they are missing; exits with a hint when that fails."""
    try:
        import pydantic
        import yaml
    except ImportError:
        print("[JINX BOOTSTRAP] Missing dependencies. Installing pydantic and pyyaml...", file=sys.stderr)
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic>=2.0.0", "pyyaml>=6.0"], stdout=subprocess.DEVNULL)
            print("[JINX BOOTSTRAP] Dependencies installed successfully.", file=sys.stderr)
        except Exception as e:
            print(f"[JINX BOOTSTRAP ERROR] Automatic dependency installation failed: {e}", file=sys.stderr)
            print('Please run: "%s" -m pip install "pydantic>=2.0.0" "pyyaml>=6.0"' % sys.executable, file=sys.stderr)
            print("If pip says the environment is externally managed, use a virtual environment: python -m venv .venv, "
                  "activate it, install there, and start the editor from that shell.", file=sys.stderr)
            sys.exit(1)

def _utf8_output():
    """Print as UTF-8, and never fail on a character that cannot be printed.

    A pipe (which is what an editor reads) gets the ANSI code page on Windows, cp1251 or cp1252: a Cyrillic task or file
    name then reaches an editor that reads UTF-8 as garbage, and an arrow or an emoji in a name or a diff (``--peek``,
    ``--export``) raises UnicodeEncodeError instead of being printed. The same goes for the bytes of a POSIX file name that
    are not UTF-8: they are printed as U+FFFD instead of ending the run.
    """
    for stream, errors in ((sys.stdout, "replace"), (sys.stderr, None)):
        try:
            if errors:
                stream.reconfigure(encoding="utf-8", errors=errors)
            else:
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass                              # no such stream (pythonw), or one that cannot be changed

_utf8_output()

# Check and install dependencies before importing JINX package modules
bootstrap_dependencies()

# Looking at the other sessions (--peek) changes nothing, so it must not run the self-patch preflight (which
# could judge another session's edit in progress) and does not need the runner either. Collecting the rounds
# into one .mbox (--export) reads the round files and writes nothing but its own output (.agent/export/).
# Closing sessions (--close) touches only their own files and cards, never JINX's source, so it needs neither.
if "--peek" in sys.argv[1:]:
    from jinx.team import peek_main
    sys.exit(_code(peek_main(sys.argv[1:], src_path.parent)))
if "--export" in sys.argv[1:]:
    from jinx.export import export_main
    sys.exit(_code(export_main(sys.argv[1:], src_path.parent)))
if "--close" in sys.argv[1:]:
    from jinx.scope import close_main
    sys.exit(_code(close_main(sys.argv[1:], src_path.parent)))

# Revert a broken self-edit before importing the package, not after.
self_patch_preflight()

try:
    from jinx.cli import main
except ImportError as e:
    print(f"[JINX BOOTSTRAP ERROR] Failed to load JINX modules: {e}", file=sys.stderr)
    sys.exit(_code(1))

if __name__ == "__main__":
    if _VIA:
        try:
            main()
        except SystemExit as e:
            sys.exit(_code(e.code))
    else:
        main()
