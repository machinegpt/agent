from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple


StartFn = Callable[["PluginContext"], Awaitable[None]] | Callable[["PluginContext"], None] | Callable[[], Awaitable[None]] | Callable[[], None]
StopFn = Callable[["PluginContext"], Awaitable[None]] | Callable[["PluginContext"], None] | Callable[[], Awaitable[None]] | Callable[[], None]
EventCallback = Callable[[str, Any], Awaitable[None]] | Callable[[str, Any], None]


@dataclass
class Plugin:
    name: str
    start: Optional[StartFn] = None
    stop: Optional[StopFn] = None
    enabled: bool = True
    priority: int = 100  # lower starts earlier
    version: str = "0.0.0"
    deps: List[str] = field(default_factory=list)
    features: Set[str] = field(default_factory=set)
    # internal
    _status: str = field(default="idle", init=False)
    _error: str | None = field(default=None, init=False)


@dataclass
class PluginContext:
    loop: asyncio.AbstractEventLoop
    shutdown_event: Any
    settings: Any | None = None
    publish: Callable[[str, Any], None] | None = None


_REGISTRY: Dict[str, Plugin] = {}
_EVENTS: Dict[str, List[Tuple[str, EventCallback]]] = {}
_CTX: PluginContext | None = None


def set_plugin_context(ctx: PluginContext) -> None:
    global _CTX
    _CTX = ctx


def _apply_env_overrides() -> None:
    return


def register_plugin(
    name: str,
    *,
    start: Optional[StartFn] = None,
    stop: Optional[StopFn] = None,
    enabled: bool = True,
    priority: int = 100,
    version: str = "0.0.0",
    deps: Optional[List[str]] = None,
    features: Optional[Set[str]] = None,
) -> None:
    if not name:
        raise ValueError("plugin name required")
    _REGISTRY[name] = Plugin(
        name=name,
        start=start,
        stop=stop,
        enabled=enabled,
        priority=priority,
        version=version,
        deps=list(deps or []),
        features=set(features or set()),
    )
    _apply_env_overrides()


def enable_plugin(name: str) -> None:
    if name in _REGISTRY:
        _REGISTRY[name].enabled = True


def disable_plugin(name: str) -> None:
    if name in _REGISTRY:
        _REGISTRY[name].enabled = False


def list_plugins() -> List[Plugin]:
    return sorted(_REGISTRY.values(), key=lambda p: (p.priority, p.name))


def subscribe_event(topic: str, *, plugin: str, callback: EventCallback) -> None:
    topic = (topic or "").strip()
    if not topic:
        return
    _EVENTS.setdefault(topic, []).append((plugin, callback))


def publish_event(topic: str, payload: Any) -> None:
    cbs = list(_EVENTS.get(topic, []))
    if not cbs:
        return
    loop = (_CTX.loop if _CTX else asyncio.get_event_loop())
    budget_ms = 60
    async def _fire() -> None:
        sem = asyncio.Semaphore(4)
        tasks: List[asyncio.Task] = []
        async def _one(cb: EventCallback) -> None:
            async with sem:
                try:
                    res = cb(topic, payload)
                    if asyncio.iscoroutine(res):
                        await asyncio.wait_for(res, timeout=budget_ms / 1000.0)  # type: ignore[arg-type]
                except Exception:
                    pass
        for _name, cb in cbs:
            tasks.append(asyncio.create_task(_one(cb)))
        if tasks:
            with contextlib.suppress(asyncio.CancelledError):  # pragma: no cover - ensure cancellation safety
                await asyncio.gather(*tasks, return_exceptions=True)
    loop.create_task(_fire())


async def _call_with_ctx(fn: Optional[StartFn | StopFn], ctx: PluginContext, timeout_ms: int) -> None:
    if not fn:
        return
    try:
        res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
    except Exception:
        return
    if asyncio.iscoroutine(res):
        try:
            await asyncio.wait_for(res, timeout=timeout_ms / 1000.0)  # type: ignore[arg-type]
        except Exception:
            pass


def _toposort_enabled() -> List[Plugin]:
    enabled = [p for p in _REGISTRY.values() if p.enabled]
    by_name = {p.name: p for p in enabled}
    indeg: Dict[str, int] = {p.name: 0 for p in enabled}
    adj: Dict[str, List[str]] = {p.name: [] for p in enabled}
    for p in enabled:
        for d in (p.deps or []):
            if d in by_name:
                indeg[p.name] += 1
                adj[d].append(p.name)
    # Kahn's algorithm with priority tie-breaker
    ready = [p for p in enabled if indeg[p.name] == 0]
    ready.sort(key=lambda x: (x.priority, x.name))
    order: List[Plugin] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for nxt in adj.get(cur.name, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(by_name[nxt])
                ready.sort(key=lambda x: (x.priority, x.name))
    # If cycle, fall back to priority order
    if len(order) != len(enabled):
        order = sorted(enabled, key=lambda p: (p.priority, p.name))
    return order


async def start_plugins() -> None:
    """Start enabled plugins with dependency-aware parallelization and budgets."""
    if _CTX is None:
        # Create a minimal best-effort context
        loop = asyncio.get_running_loop()
        ctx = PluginContext(loop=loop, shutdown_event=asyncio.Event(), settings=None, publish=publish_event)
    else:
        ctx = _CTX
    start_ms = 400
    conc = 3
    sem = asyncio.Semaphore(max(1, conc))
    order = _toposort_enabled()
    tasks: List[asyncio.Task] = []

    async def _run(p: Plugin) -> None:
        async with sem:
            p._status = "starting"
            try:
                await _call_with_ctx(p.start, ctx, start_ms)
                p._status = "running"
                p._error = None
            except Exception as e:
                p._status = "error"
                p._error = str(e)

    for p in order:
        tasks.append(asyncio.create_task(_run(p)))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def stop_plugins() -> None:
    if _CTX is None:
        loop = asyncio.get_running_loop()
        ctx = PluginContext(loop=loop, shutdown_event=asyncio.Event(), settings=None, publish=publish_event)
    else:
        ctx = _CTX
    stop_ms = 300
    conc = 3
    sem = asyncio.Semaphore(max(1, conc))
    order = [p for p in _toposort_enabled()][::-1]
    tasks: List[asyncio.Task] = []

    async def _run(p: Plugin) -> None:
        async with sem:
            try:
                p._status = "stopping"
                await _call_with_ctx(p.stop, ctx, stop_ms)
                p._status = "idle"
            except Exception:
                pass

    for p in order:
        tasks.append(asyncio.create_task(_run(p)))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def reload_plugin(name: str) -> None:
    p = _REGISTRY.get(name)
    if not p or not p.enabled:
        return
    if _CTX is None:
        loop = asyncio.get_running_loop()
        ctx = PluginContext(loop=loop, shutdown_event=asyncio.Event(), settings=None, publish=publish_event)
    else:
        ctx = _CTX
    stop_ms = 300
    start_ms = 400
    try:
        await _call_with_ctx(p.stop, ctx, stop_ms)
    finally:
        await _call_with_ctx(p.start, ctx, start_ms)


async def start_plugin(name: str) -> None:
    p = _REGISTRY.get(name)
    if not p or not p.enabled:
        return
    if p._status == "running":
        return
    if _CTX is None:
        loop = asyncio.get_running_loop()
        ctx = PluginContext(loop=loop, shutdown_event=asyncio.Event(), settings=None, publish=publish_event)
    else:
        ctx = _CTX
    start_ms = 400
    await _call_with_ctx(p.start, ctx, start_ms)
    p._status = "running"
    p._error = None


async def stop_plugin(name: str) -> None:
    p = _REGISTRY.get(name)
    if not p:
        return
    if _CTX is None:
        loop = asyncio.get_running_loop()
        ctx = PluginContext(loop=loop, shutdown_event=asyncio.Event(), settings=None, publish=publish_event)
    else:
        ctx = _CTX
    stop_ms = 300
    await _call_with_ctx(p.stop, ctx, stop_ms)
    p._status = "idle"

__all__ = [
    "Plugin",
    "PluginContext",
    "set_plugin_context",
    "register_plugin",
    "enable_plugin",
    "disable_plugin",
    "list_plugins",
    "subscribe_event",
    "publish_event",
    "start_plugins",
    "stop_plugins",
    "reload_plugin",
    "start_plugin",
    "stop_plugin",
]
