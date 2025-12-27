from __future__ import annotations

import sys
import time
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, field

__all__ = ["boot", "KernelState", "get_kernel_state", "register_boot_hook", "BootPhase"]


class BootPhase:
    RESILIENCE = "resilience"
    ENV = "env"
    STARTUP_CHECKS = "startup_checks"
    PREWARM = "prewarm"
    OTEL = "otel"
    HOOKS = "hooks"


@dataclass(slots=True)
class KernelState:
    boot_time_ns: int = 0
    phases_completed: Dict[str, float] = field(default_factory=dict)
    boot_hooks_executed: int = 0
    resilience_installed: bool = False
    otel_enabled: bool = False
    prewarm_done: bool = False


_kernel_state: Optional[KernelState] = None
_boot_hooks: list[Callable[[], None]] = []


def get_kernel_state() -> Optional[KernelState]:
    return _kernel_state


def register_boot_hook(hook: Callable[[], None]) -> None:
    _boot_hooks.append(hook)


def _phase(state: KernelState, name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    def decorator(fn: Callable[[], Any]) -> Callable[[], Any]:
        def wrapper() -> Any:
            t0 = time.perf_counter()
            result = fn()
            state.phases_completed[name] = (time.perf_counter() - t0) * 1000
            return result
        return wrapper
    return decorator


def boot() -> None:
    global _kernel_state
    _kernel_state = KernelState(boot_time_ns=time.perf_counter_ns())

    @_phase(_kernel_state, BootPhase.RESILIENCE)
    def _install_resilience() -> None:
        from jinx.micro.runtime.resilience import install_resilience
        install_resilience()
        _kernel_state.resilience_installed = True

    @_phase(_kernel_state, BootPhase.STARTUP_CHECKS)
    def _startup_checks() -> None:
        from jinx.micro.runtime.startup_checks import run_startup_checks
        run_startup_checks(stage="pre")

    @_phase(_kernel_state, BootPhase.PREWARM)
    def _prewarm() -> None:
        from jinx.micro.net.client import prewarm_openai_client
        prewarm_openai_client()
        _kernel_state.prewarm_done = True

    @_phase(_kernel_state, BootPhase.OTEL)
    def _setup_otel() -> None:
        from jinx.observability.setup import setup_otel
        setup_otel()
        _kernel_state.otel_enabled = True

    @_phase(_kernel_state, BootPhase.HOOKS)
    def _run_hooks() -> None:
        for hook in _boot_hooks:
            hook()
            _kernel_state.boot_hooks_executed += 1

    for phase_fn in [_install_resilience, _startup_checks, _prewarm, _setup_otel, _run_hooks]:
        try:
            phase_fn()
        except ImportError:
            pass
        except Exception:
            pass
