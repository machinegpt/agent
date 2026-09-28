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

    ``prompts`` is a leaf module of pure string constants and is loaded by the
    import machinery here, so the brake still does not depend on the code it
    protects being importable.

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
    spec.loader.exec_module(mod)
    return mod


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
    except Exception as e:  # a corrupt selfpatch.py itself: nothing we can do here
        print("[JINX SELF-PATCH] Preflight unavailable: %s" % e, file=sys.stderr)
        return
    if mod is None:
        return

    try:
        if not mod.BASELINE_DIR.exists():
            return
        changed = mod.baseline_changed()
        if not changed:
            # Nothing was touched, so the baseline is still the right reference
            # for the rest of the run. Clearing it here would disarm the gate
            # before the first tool call ever happened.
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
        # Same rule as the runner: the suite that judges the patch is itself part
        # of what the patch could have edited, so it is put back before it runs.
        # The result is kept and reported: the runner already tells the model
        # which of its test edits were rolled back, and a bootstrap that
        # discarded the same files without a word left the model believing
        # tests it had just written were still in place.
        tests_restored = mod.restore_baseline(only=mod.REPO_PREFIX)
        result = mod.verify(src_path.parent.parent)
        if result["ok"]:
            if tests_restored:
                _record_feedback(
                    "[JINX SELF-PATCH] The test edits from the previous run were "
                    "rolled back before verification, because the suite that "
                    "judges a self-patch is itself part of what a self-patch can "
                    "edit. Reverted: %s. The source edits were verified and kept; "
                    "re-add the tests if they are still needed."
                    % ", ".join(tests_restored)
                )
            print("[JINX SELF-PATCH] Verified: %s" % ", ".join(changed), file=sys.stderr)
            # The edit passed, so adopt it as the new reference instead of
            # re-verifying the same diff on every subsequent process.
            mod.capture_baseline()
            return
        restored = mod.restore_baseline()
    except Exception as e:
        print("[JINX SELF-PATCH] Preflight failed: %s" % e, file=sys.stderr)
        return

    message = (
        "[JINX SELF-PATCH REVERTED] The previous round edited JINX's own source "
        "(%s) and broke it, so the framework could not even start. The change was "
        "rolled back automatically and the run continues (%s).\n%s"
        % (
            ", ".join(changed),
            result["summary"],
            "\n".join("[%s] %s" % (c["name"], c["tail"]) for c in result["checks"]),
        )
    )
    print(message, file=sys.stderr)
    _record_feedback(message)


def _record_feedback(message):
    """Hands a preflight verdict to the next prompt via the run state.

    pyyaml is an external dependency, not part of the package, so this still
    works when the source tree itself is too broken to import JINX.
    """
    try:
        import yaml
        run_state_path = Path(__file__).resolve().parent / "jinx_run_state.yaml"
        if run_state_path.is_file():
            data = yaml.safe_load(run_state_path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                data["self_patch_feedback"] = message
                run_state_path.write_text(
                    yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
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

# Revert a broken self-edit before importing the package, not after.
self_patch_preflight()

try:
    from jinx.cli import main
except ImportError as e:
    print(f"[JINX BOOTSTRAP ERROR] Failed to load JINX modules: {e}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
