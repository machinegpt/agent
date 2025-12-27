from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Dict, Optional

from jinx.settings import Settings


class JobHealth(IntEnum):
    HEALTHY = 0
    DEGRADED = 1
    FAILING = 2
    DEAD = 3


@dataclass(slots=True)
class JobMetrics:
    start_count: int = 0
    failure_count: int = 0
    success_count: int = 0
    last_start_time: float = 0.0
    last_failure_time: float = 0.0
    total_runtime_s: float = 0.0
    health: JobHealth = JobHealth.HEALTHY
    consecutive_failures: int = 0


@dataclass(slots=True)
class SupervisedJob:
    name: str
    start: Callable[[], "asyncio.Task[None]"]
    critical: bool = False
    health_check: Optional[Callable[[], bool]] = None
    max_restart_rate: float = 5.0


async def _sleep_cancelable(delay: float, cancel_event: asyncio.Event) -> None:
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=delay)
    except asyncio.TimeoutError:
        pass


@dataclass(slots=True)
class RecoveryContext:
    task_name: str
    exception: BaseException
    restart_count: int
    consecutive_failures: int
    cooldowns: Dict[str, float] = field(default_factory=dict)

    def cooldown_ok(self, key: str, cd_s: float) -> bool:
        now = time.time()
        if (now - self.cooldowns.get(key, 0.0)) >= cd_s:
            self.cooldowns[key] = now
            return True
        return False


class RecoveryPipeline:
    __slots__ = ("_handlers",)

    def __init__(self) -> None:
        self._handlers: list[Callable[[RecoveryContext], Awaitable[None]]] = []

    def register(self, handler: Callable[[RecoveryContext], Awaitable[None]]) -> None:
        self._handlers.append(handler)

    async def execute(self, ctx: RecoveryContext) -> None:
        for handler in self._handlers:
            await handler(ctx)


def _create_default_pipeline() -> RecoveryPipeline:
    pipeline = RecoveryPipeline()

    async def _log_handler(ctx: RecoveryContext) -> None:
        from jinx.logging_service import bomb_log
        await bomb_log(f"Recovery: {ctx.task_name} - {type(ctx.exception).__name__}: {ctx.exception}")

    async def _heal_handler(ctx: RecoveryContext) -> None:
        if not ctx.cooldown_ok("heal", 3.0):
            return
        import traceback
        tb = "".join(traceback.format_exception(type(ctx.exception), ctx.exception, ctx.exception.__traceback__))
        from jinx.micro.runtime.self_healing import auto_heal_error
        await auto_heal_error(type(ctx.exception).__name__, str(ctx.exception), tb)

    async def _repairs_handler(ctx: RecoveryContext) -> None:
        if not ctx.cooldown_ok("repairs", 15.0):
            return
        from jinx.micro.runtime.resilience import schedule_repairs
        await schedule_repairs()

    pipeline.register(_log_handler)
    pipeline.register(_heal_handler)
    pipeline.register(_repairs_handler)
    return pipeline


_recovery_pipeline = _create_default_pipeline()


class Supervisor:
    __slots__ = ("_jobs", "_shutdown", "_settings", "_tasks", "_metrics", "_restarts", "_restart_times")

    def __init__(self, jobs: list[SupervisedJob], shutdown_event: asyncio.Event, settings: Settings) -> None:
        self._jobs = {j.name: j for j in jobs}
        self._shutdown = shutdown_event
        self._settings = settings
        self._tasks: Dict[str, asyncio.Task[None]] = {}
        self._metrics: Dict[str, JobMetrics] = {j.name: JobMetrics() for j in jobs}
        self._restarts: Dict[str, int] = {}
        self._restart_times: Dict[str, list[float]] = {j.name: [] for j in jobs}

    def _start_job(self, name: str) -> None:
        job = self._jobs.get(name)
        if not job:
            return
        task = job.start()
        self._tasks[name] = task
        self._metrics[name].start_count += 1
        self._metrics[name].last_start_time = time.time()

    def _compute_backoff(self, name: str, count: int) -> float:
        rt = self._settings.runtime
        m = self._metrics[name]
        base = max(1, rt.backoff_min_ms) / 1000.0
        cap = max(base, rt.backoff_max_ms / 1000.0)
        delay = min(cap, base * (2 ** count))
        delay *= 0.7 + 0.6 * random.random()
        if m.health == JobHealth.FAILING:
            delay *= 2.0
        return delay

    async def _handle_failure(self, name: str, ex: BaseException) -> None:
        rt = self._settings.runtime
        m = self._metrics[name]
        count = self._restarts.get(name, 0)

        ctx = RecoveryContext(
            task_name=name,
            exception=ex,
            restart_count=count,
            consecutive_failures=m.consecutive_failures,
        )
        asyncio.create_task(_recovery_pipeline.execute(ctx))

        if not rt.supervise_tasks:
            return

        if count >= rt.autorestart_limit:
            m.health = JobHealth.DEAD
            job = self._jobs.get(name)
            if job and job.critical:
                from jinx.logging_service import bomb_log
                await bomb_log(f"CRITICAL JOB DEAD: {name}")
                self._shutdown.set()
            return

        now = time.time()
        self._restart_times[name] = [t for t in self._restart_times[name] if (now - t) < 60.0]
        self._restart_times[name].append(now)
        restart_rate = len(self._restart_times[name])

        job = self._jobs.get(name)
        if job and restart_rate > job.max_restart_rate:
            m.health = JobHealth.FAILING
        elif m.consecutive_failures >= 2:
            m.health = JobHealth.DEGRADED

        m.consecutive_failures = count
        self._restarts[name] = count + 1

        delay = self._compute_backoff(name, count)
        await _sleep_cancelable(delay, self._shutdown)

        if not self._shutdown.is_set():
            self._start_job(name)

    async def run(self) -> None:
        for name in self._jobs:
            self._start_job(name)

        while not self._shutdown.is_set():
            if not self._tasks:
                await self._shutdown.wait()
                break

            shutdown_task = asyncio.create_task(self._shutdown.wait())
            done, _ = await asyncio.wait(
                set(self._tasks.values()) | {shutdown_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not shutdown_task.done():
                shutdown_task.cancel()

            if self._shutdown.is_set():
                break

            for task in done:
                name = next((k for k, v in self._tasks.items() if v is task), None)
                if not name:
                    continue

                self._tasks.pop(name, None)
                ex = task.exception() if not task.cancelled() else None

                if ex is None:
                    continue

                await self._handle_failure(name, ex)

        for task in self._tasks.values():
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)


async def run_supervisor(jobs: list[SupervisedJob], shutdown_event: asyncio.Event, settings: Settings) -> None:
    supervisor = Supervisor(jobs, shutdown_event, settings)
    await supervisor.run()
