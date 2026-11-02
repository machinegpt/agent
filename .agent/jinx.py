#!/usr/bin/env python3
"""
Direct entry point for JINX.
Designed to be executed when the repository is dropped into a project as a `.agent` folder.

Usage:
  python .agent/jinx.py "Analyze and fix the bug in auth"
"""
import sys
from pathlib import Path

# Insert the local src/ directory into sys.path so 'import jinx' resolves correctly
# without needing a global pip installation.
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))


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
            print("Please manually run: pip install pydantic>=2.0.0 pyyaml>=6.0", file=sys.stderr)
            sys.exit(1)

# Check and install dependencies before importing JINX package modules
bootstrap_dependencies()

# Looking at the other sessions (--peek) changes nothing, so it must not run the self-patch preflight (which
# could judge another session's edit in progress) and does not need the runner either. Collecting the rounds
# into one .mbox (--export) reads the round files and writes nothing but its own output (.agent/export/).
if "--peek" in sys.argv[1:]:
    from jinx.team import peek_main
    sys.exit(peek_main(sys.argv[1:], src_path.parent))
if "--export" in sys.argv[1:]:
    from jinx.export import export_main
    sys.exit(export_main(sys.argv[1:], src_path.parent))

# Revert a broken self-edit before importing the package, not after.
self_patch_preflight()

try:
    from jinx.cli import main
except ImportError as e:
    print(f"[JINX BOOTSTRAP ERROR] Failed to load JINX modules: {e}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
