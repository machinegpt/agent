from __future__ import annotations

import os
import sys
import traceback
import types
from pathlib import Path


class PackageResolver:
    __slots__ = ("_script_dir",)

    def __init__(self, script_path: str) -> None:
        self._script_dir = Path(script_path).parent

    def resolve(self) -> None:
        pkg_dir = self._script_dir / "jinx"
        override = os.getenv("JINX_PACKAGE_DIR", "").strip()
        if override and Path(override).is_dir():
            pkg_dir = Path(override)

        if pkg_dir.is_dir() and "jinx" not in sys.modules:
            pkg = types.ModuleType("jinx")
            pkg.__path__ = [str(pkg_dir)]
            sys.modules["jinx"] = pkg


class CognitiveRepairBootstrap:
    __slots__ = ("_max_attempts", "_attempt")

    def __init__(self, max_attempts: int = 5) -> None:
        self._max_attempts = max_attempts
        self._attempt = 0

    def _try_cognitive_repair(self, exc: Exception, tb_str: str) -> bool:
        try:
            from jinx.micro.runtime.repair_agent import auto_repair_and_restart
            return auto_repair_and_restart(exc, [__file__])
        except ImportError as e:
            print(f"Repair agent not available: {e}", file=sys.stderr)
            return False
        except Exception as repair_error:
            print(f"Cognitive repair failed: {repair_error}", file=sys.stderr)
            return False

    def run(self, fn: callable) -> int:
        while self._attempt < self._max_attempts:
            try:
                return fn()
            except KeyboardInterrupt:
                return 130
            except Exception as exc:
                self._attempt += 1
                tb = traceback.format_exc()
                
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"JINX ERROR (attempt {self._attempt}/{self._max_attempts})", file=sys.stderr)
                print(f"{'='*60}", file=sys.stderr)
                print(tb, file=sys.stderr)
                
                if self._attempt < self._max_attempts:
                    if self._try_cognitive_repair(exc, tb):
                        continue
                
                print(f"Fatal: {exc}", file=sys.stderr)
                return 1
        return 1


def _boot_and_run() -> int:
    PackageResolver(__file__).resolve()

    from jinx.kernel import boot
    boot()

    from jinx.orchestrator import main
    main()
    return 0


def main() -> int:
    if os.getenv("JINX_NO_REPAIR", "").lower() in ("1", "true", "yes"):
        try:
            return _boot_and_run()
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"Fatal: {exc}", file=sys.stderr)
            return 1
    
    return CognitiveRepairBootstrap().run(_boot_and_run)


if __name__ == "__main__":
    sys.exit(main())
