from __future__ import annotations

import asyncio
import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import jinx.state as jx_state
from jinx.banner_service import show_banner
from jinx.utils import chaos_patch
from jinx.runtime import start_input_task, frame_shift
from jinx.embeddings import (
    start_embeddings_task,
    start_project_embeddings_task,
    stop_embeddings_task,
    stop_project_embeddings_task,
)
from jinx.memory.optimizer import stop as stop_memory_optimizer, start_memory_optimizer_task
from jinx.conversation.error_worker import stop_error_worker
from jinx.settings import Settings
from jinx.supervisor import run_supervisor, SupervisedJob
from jinx.priority import start_priority_dispatcher_task
from jinx.autotune import start_autotune_task
from jinx.watchdog import start_watchdog_task
from jinx.micro.embeddings.retrieval_core import shutdown_proc_pool
from jinx.micro.net.client import prewarm_openai_client
from jinx.micro.runtime.api import stop_selfstudy, ensure_runtime
from jinx.micro.runtime.plugins import (
    start_plugins,
    stop_plugins,
    set_plugin_context,
    PluginContext,
    publish_event,
)
from jinx.micro.runtime.builtin_plugins import register_builtin_plugins
from jinx.micro.runtime.self_update_handshake import set_online, set_healthy


@dataclass(slots=True)
class RuntimeContext:
    settings: Settings
    q_in: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    q_proc: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        maxsize = self.settings.runtime.queue_maxsize
        self.q_in = asyncio.Queue(maxsize=maxsize)
        self.q_proc = asyncio.Queue(maxsize=maxsize)


class ShutdownManager:
    __slots__ = ("_stop_fns",)

    def __init__(self) -> None:
        self._stop_fns: List[Callable] = []

    def register(self, fn: Callable) -> None:
        self._stop_fns.append(fn)

    async def execute(self) -> None:
        for fn in self._stop_fns:
            await fn()


def _build_job_specs(ctx: RuntimeContext) -> List[SupervisedJob]:
    return [
        SupervisedJob(name="input", start=lambda: start_input_task(ctx.q_in)),
        SupervisedJob(name="frame", start=lambda: asyncio.create_task(frame_shift(ctx.q_proc))),
        SupervisedJob(name="priority", start=lambda: start_priority_dispatcher_task(ctx.q_in, ctx.q_proc, ctx.settings)),
        SupervisedJob(name="embeddings", start=start_embeddings_task),
        SupervisedJob(name="memopt", start=start_memory_optimizer_task),
        SupervisedJob(name="proj-embed", start=start_project_embeddings_task),
        SupervisedJob(name="autotune", start=lambda: start_autotune_task(ctx.q_in, ctx.settings)),
        SupervisedJob(name="watchdog", start=lambda: start_watchdog_task(ctx.settings)),
    ]


async def _setup_plugins(ctx: RuntimeContext) -> None:
    loop = asyncio.get_running_loop()
    set_plugin_context(PluginContext(
        loop=loop,
        shutdown_event=jx_state.shutdown_event,
        settings=ctx.settings,
        publish=publish_event,
    ))
    register_builtin_plugins()
    await start_plugins()


async def _cleanup(shutdown_mgr: ShutdownManager) -> None:
    await shutdown_mgr.execute()

    loop = asyncio.get_running_loop()
    pending = [t for t in asyncio.all_tasks(loop) if not t.done() and t != asyncio.current_task()]

    for task in pending:
        task.cancel()

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    shutdown_proc_pool()


async def pulse_core(settings: Optional[Settings] = None) -> None:
    show_banner()

    cfg = settings or Settings()
    cfg.apply_to_state()

    ctx = RuntimeContext(settings=cfg)

    from jinx.observability.recorder import start_run_record, finalize_run_record
    ctx.run_id = start_run_record({"settings": cfg.to_dict()})

    print(
        f"‖ Auto-tune: prio={'on' if cfg.runtime.use_priority_queue else 'off'}, "
        f"threads={cfg.runtime.threads_max_workers}, "
        f"queue={cfg.runtime.queue_maxsize}, rt={cfg.runtime.hard_rt_budget_ms}ms"
    )

    prewarm_openai_client()
    await ensure_runtime()
    set_online()

    shutdown_mgr = ShutdownManager()
    shutdown_mgr.register(stop_error_worker)
    shutdown_mgr.register(stop_memory_optimizer)
    shutdown_mgr.register(stop_embeddings_task)
    shutdown_mgr.register(stop_project_embeddings_task)
    shutdown_mgr.register(stop_selfstudy)
    shutdown_mgr.register(stop_plugins)

    async with chaos_patch():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(cf.ThreadPoolExecutor(
            max_workers=cfg.runtime.threads_max_workers,
            thread_name_prefix="jinx-worker",
        ))

        job_specs = _build_job_specs(ctx)

        from jinx.micro.runtime.health_monitor import start_health_monitoring

        async def _health_loop() -> None:
            await start_health_monitoring()
            while not jx_state.shutdown_event.is_set():
                await asyncio.sleep(1.0)

        job_specs.append(SupervisedJob(name="health-monitor", start=lambda: asyncio.create_task(_health_loop())))

        await _setup_plugins(ctx)
        set_healthy()
        await asyncio.sleep(0.05)

        try:
            await run_supervisor(job_specs, jx_state.shutdown_event, cfg)
        finally:
            await _cleanup(shutdown_mgr)
            if ctx.run_id:
                finalize_run_record(ctx.run_id, extra={"status": "shutdown"})
