from __future__ import annotations

import asyncio
import time
import threading
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(slots=True, frozen=True)
class StateChangeEvent:
    key: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=time.perf_counter)
    source: str = "unknown"


class EventBus(Generic[T]):
    __slots__ = ("_handlers", "_lock", "_async_handlers")

    def __init__(self) -> None:
        self._handlers: list[Callable[[T], None]] = []
        self._async_handlers: list[Callable[[T], Awaitable[None]]] = []
        self._lock = threading.RLock()

    def subscribe(self, handler: Callable[[T], None]) -> Callable[[], None]:
        with self._lock:
            self._handlers.append(handler)
        return lambda: self._unsubscribe(handler)

    def subscribe_async(self, handler: Callable[[T], Awaitable[None]]) -> Callable[[], None]:
        with self._lock:
            self._async_handlers.append(handler)
        return lambda: self._unsubscribe_async(handler)

    def _unsubscribe(self, handler: Callable[[T], None]) -> None:
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def _unsubscribe_async(self, handler: Callable[[T], Awaitable[None]]) -> None:
        with self._lock:
            if handler in self._async_handlers:
                self._async_handlers.remove(handler)

    def publish(self, event: T) -> None:
        with self._lock:
            handlers = list(self._handlers)
        for h in handlers:
            h(event)

    async def publish_async(self, event: T) -> None:
        with self._lock:
            handlers = list(self._async_handlers)
        for h in handlers:
            await h(event)


class ReactiveVar(Generic[T]):
    __slots__ = ("_value", "_lock", "_observers")

    def __init__(self, initial: T) -> None:
        self._value = initial
        self._lock = threading.RLock()
        self._observers: list[Callable[[T, T], None]] = []

    def get(self) -> T:
        with self._lock:
            return self._value

    def set(self, value: T, notify: bool = True) -> None:
        with self._lock:
            old = self._value
            self._value = value
            if notify and old != value:
                for obs in list(self._observers):
                    obs(old, value)

    def subscribe(self, callback: Callable[[T, T], None]) -> Callable[[], None]:
        with self._lock:
            self._observers.append(callback)
        return lambda: self._observers.remove(callback) if callback in self._observers else None

    def __repr__(self) -> str:
        return f"ReactiveVar({self._value!r})"


class LoopBoundLock:
    __slots__ = ("_locks",)

    def __init__(self) -> None:
        self._locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()

    def _get(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        if loop not in self._locks:
            self._locks[loop] = asyncio.Lock()
        return self._locks[loop]

    async def acquire(self) -> bool:
        return await self._get().acquire()

    def release(self) -> None:
        self._get().release()

    def locked(self) -> bool:
        return self._get().locked()

    async def __aenter__(self) -> "LoopBoundLock":
        await self.acquire()
        return self

    async def __aexit__(self, *_) -> bool:
        self.release()
        return False


class LoopBoundEvent:
    __slots__ = ("_events", "_flag")

    def __init__(self) -> None:
        self._events: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Event] = weakref.WeakKeyDictionary()
        self._flag = False

    def _get(self) -> asyncio.Event:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        if loop not in self._events:
            ev = asyncio.Event()
            if self._flag:
                ev.set()
            self._events[loop] = ev
        return self._events[loop]

    def is_set(self) -> bool:
        return self._flag

    def set(self) -> None:
        self._flag = True
        for loop, ev in list(self._events.items()):
            if loop.is_running():
                loop.call_soon_threadsafe(ev.set)
            else:
                ev.set()

    def clear(self) -> None:
        self._flag = False
        for loop, ev in list(self._events.items()):
            if loop.is_running():
                loop.call_soon_threadsafe(ev.clear)
            else:
                ev.clear()

    async def wait(self) -> bool:
        if self._flag:
            return True
        await self._get().wait()
        return True


shard_lock: Any = LoopBoundLock()
_sync_lock = threading.RLock()
_state_bus: EventBus[StateChangeEvent] = EventBus()

pulse: int = 100
boom_limit: int = 30
activity: str = ""
activity_ts: float = 0.0
activity_detail: Optional[Dict[str, Any]] = None
activity_detail_ts: float = 0.0
last_agent_reply_ts: float = 0.0
ui_prompt_toolkit: bool = False
ui_input_active: bool = False
ui_output_active: bool = False
state_compiler_llm_on: bool = False

shutdown_event: Any = LoopBoundEvent()
throttle_event: Any = LoopBoundEvent()


def register_state_observer(callback: Callable[[StateChangeEvent], None]) -> Callable[[], None]:
    return _state_bus.subscribe(callback)


def unregister_state_observer(callback: Callable[[StateChangeEvent], None]) -> None:
    _state_bus._unsubscribe(callback)


def _notify(key: str, old: Any, new: Any, source: str = "unknown") -> None:
    _state_bus.publish(StateChangeEvent(key=key, old_value=old, new_value=new, source=source))


def set_activity(new_activity: str, detail: Optional[Dict[str, Any]] = None) -> None:
    global activity, activity_ts, activity_detail, activity_detail_ts
    with _sync_lock:
        old = activity
        activity = new_activity
        activity_ts = time.perf_counter()
        if detail is not None:
            activity_detail = detail
            activity_detail_ts = activity_ts
        _notify("activity", old, new_activity, "set_activity")


def atomic_pulse_decrement(amount: int = 1) -> int:
    global pulse
    with _sync_lock:
        old = pulse
        pulse = max(0, pulse - amount)
        if old != pulse:
            _notify("pulse", old, pulse, "atomic_pulse_decrement")
        return pulse


def get_state_snapshot() -> Dict[str, Any]:
    with _sync_lock:
        return {
            "pulse": pulse,
            "boom_limit": boom_limit,
            "activity": activity,
            "activity_ts": activity_ts,
            "shutdown": shutdown_event.is_set(),
            "throttle": throttle_event.is_set(),
        }


ctx_current_operation: ContextVar[str] = ContextVar("current_operation", default="")
ctx_priority_level: ContextVar[int] = ContextVar("priority_level", default=Priority.NORMAL)
