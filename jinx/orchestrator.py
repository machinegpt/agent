from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional
import time

from jinx.bootstrap import load_env


@dataclass(slots=True)
class OrchestratorMetrics:
    start_time_ns: int = 0
    runtime_duration_ms: float = 0.0
    shutdown_reason: str = ""
    operations: list[tuple[str, float, bool]] = field(default_factory=list)


class CrashDiagnostics:
    __slots__ = ("_record_fn", "_mark_fn", "_installed")

    def __init__(self) -> None:
        self._record_fn: Optional[Callable] = None
        self._mark_fn: Optional[Callable] = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        from jinx.micro.runtime.crash_diagnostics import (
            install_crash_diagnostics,
            record_operation,
            mark_normal_shutdown,
        )
        install_crash_diagnostics()
        self._record_fn = record_operation
        self._mark_fn = mark_normal_shutdown
        self._installed = True

    def record(self, op: str, **kwargs) -> None:
        if self._record_fn:
            self._record_fn(op, **kwargs)

    def mark(self, reason: str) -> None:
        if self._mark_fn:
            self._mark_fn(reason)


class ShutdownMonitor:
    __slots__ = ("_installed",)

    def __init__(self) -> None:
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True


def main() -> None:
    metrics = OrchestratorMetrics(start_time_ns=time.perf_counter_ns())
    diag = CrashDiagnostics()
    monitor = ShutdownMonitor()

    diag.install()
    diag.record("startup", details={"stage": "orchestrator"}, success=True)

    monitor.install()
    load_env()

    from jinx.micro.runtime.startup_checks import run_startup_checks
    run_startup_checks(stage="post")

    from jinx.runtime_service import pulse_core

    t0 = time.perf_counter()
    try:
        diag.record("runtime_start", success=True)
        asyncio.run(pulse_core())
        metrics.runtime_duration_ms = (time.perf_counter() - t0) * 1000
        metrics.shutdown_reason = "normal_completion"
        diag.record("runtime_end", success=True)
        diag.mark("normal_completion")
    except KeyboardInterrupt:
        metrics.runtime_duration_ms = (time.perf_counter() - t0) * 1000
        metrics.shutdown_reason = "keyboard_interrupt"
        diag.record("runtime_interrupted", details={"reason": "KeyboardInterrupt"}, success=True)
        diag.mark("keyboard_interrupt")
        raise
    except Exception as e:
        metrics.runtime_duration_ms = (time.perf_counter() - t0) * 1000
        metrics.shutdown_reason = f"error:{type(e).__name__}"
        diag.record("runtime_error", details={"error": str(e)}, success=False, error=str(e))
        raise
