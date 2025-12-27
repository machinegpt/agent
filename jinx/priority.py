from __future__ import annotations

import asyncio
import heapq
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Pattern, Tuple

from jinx.settings import Settings


class PriorityLevel(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(slots=True)
class PriorityMetrics:
    total_processed: int = 0
    high_priority_count: int = 0
    normal_priority_count: int = 0
    low_priority_count: int = 0
    avg_queue_time_ms: float = 0.0
    max_queue_time_ms: float = 0.0
    classification_hits: Dict[int, int] = field(default_factory=dict)
    denied_admissions: int = 0


@dataclass(slots=True, frozen=True)
class PriorityRule:
    priority: int
    pattern: str
    is_regex: bool = False
    position_limit: Optional[int] = None


class PriorityClassifier:
    __slots__ = ("_rules", "_lock")

    _instance: Optional["PriorityClassifier"] = None
    _cls_lock = threading.Lock()

    def __init__(self) -> None:
        self._rules: List[Tuple[int, Pattern[str] | str, Optional[int]]] = []
        self._lock = threading.RLock()
        self._setup_rules()

    @classmethod
    def get_instance(cls) -> "PriorityClassifier":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _setup_rules(self) -> None:
        rules = [
            (PriorityLevel.CRITICAL, r"^!!", True, None),
            (PriorityLevel.CRITICAL, r"^/urgent", True, None),
            (PriorityLevel.CRITICAL, r"^/critical", True, None),
            (PriorityLevel.CRITICAL, r"\basap\b", True, 20),
            (PriorityLevel.CRITICAL, r"\bemergency\b", True, 30),
            (PriorityLevel.HIGH, r"^![^!]", True, None),
            (PriorityLevel.HIGH, r"^/high", True, None),
            (PriorityLevel.HIGH, r"^/important", True, None),
            (PriorityLevel.HIGH, r"^/priority", True, None),
            (PriorityLevel.LOW, r"^/low", True, None),
            (PriorityLevel.LOW, r"^/defer", True, None),
            (PriorityLevel.LOW, r"^/later", True, None),
            (PriorityLevel.BACKGROUND, r"^<no_response>$", True, None),
            (PriorityLevel.BACKGROUND, r"^/background", True, None),
        ]
        for priority, pattern, is_regex, limit in rules:
            if is_regex:
                self._rules.append((priority, re.compile(pattern, re.IGNORECASE), limit))
            else:
                self._rules.append((priority, pattern.lower(), limit))

    @lru_cache(maxsize=2048)
    def classify(self, msg: str) -> int:
        if not msg:
            return PriorityLevel.NORMAL

        s = msg.strip().lower()
        if not s:
            return PriorityLevel.NORMAL

        for priority, pattern, limit in self._rules:
            text = s[:limit] if limit else s
            if isinstance(pattern, Pattern):
                if pattern.search(text):
                    return priority
            elif text.startswith(pattern):
                return priority

        return PriorityLevel.NORMAL

    def clear_cache(self) -> None:
        with self._lock:
            self.classify.cache_clear()

    def add_rule(self, rule: PriorityRule) -> None:
        with self._lock:
            if rule.is_regex:
                self._rules.append((rule.priority, re.compile(rule.pattern, re.IGNORECASE), rule.position_limit))
            else:
                self._rules.append((rule.priority, rule.pattern.lower(), rule.position_limit))
            self.clear_cache()


_classifier = PriorityClassifier.get_instance()


def classify_priority(msg: str) -> int:
    return _classifier.classify(msg)


class PriorityDispatcher:
    __slots__ = ("_src", "_dst", "_settings", "_heap", "_seq", "_metrics", "_running")

    def __init__(self, src: asyncio.Queue[str], dst: asyncio.Queue[str], settings: Settings) -> None:
        self._src = src
        self._dst = dst
        self._settings = settings
        self._heap: List[Tuple[int, int, float, str]] = []
        self._seq = 0
        self._metrics = PriorityMetrics()
        self._running = True

    def _apply_starvation_boost(self, current_time: float, threshold: float = 30.0) -> None:
        modified = False
        for i, (pr, seq, enq_t, msg) in enumerate(self._heap):
            if pr >= 2 and (current_time - enq_t) >= threshold:
                self._heap[i] = (max(0, pr - 2), seq, enq_t, msg)
                modified = True
        if modified:
            heapq.heapify(self._heap)

    async def _dispatch_one(self, loop: asyncio.AbstractEventLoop) -> bool:
        if not self._heap:
            return False

        current_time = loop.time()
        self._apply_starvation_boost(current_time)

        pr, seq, enq_t, item = heapq.heappop(self._heap)
        await self._dst.put(item)

        queue_time_ms = (current_time - enq_t) * 1000.0
        self._metrics.total_processed += 1
        self._metrics.avg_queue_time_ms = (
            (self._metrics.avg_queue_time_ms * (self._metrics.total_processed - 1) + queue_time_ms)
            / self._metrics.total_processed
        )
        self._metrics.max_queue_time_ms = max(self._metrics.max_queue_time_ms, queue_time_ms)
        return True

    def _enqueue(self, msg: str, enqueue_time: float) -> bool:
        pr = classify_priority(msg)
        capacity = max(1, self._settings.runtime.queue_maxsize)
        policy = (self._settings.runtime.queue_policy or "block").strip().lower()

        if len(self._heap) >= capacity and policy == "drop_newest":
            self._metrics.denied_admissions += 1
            return False

        if len(self._heap) >= capacity and policy == "drop_oldest" and self._heap:
            self._heap[0] = self._heap[-1]
            self._heap.pop()
            heapq.heapify(self._heap)

        heapq.heappush(self._heap, (pr, self._seq, enqueue_time, msg))
        self._seq += 1
        self._metrics.classification_hits[pr] = self._metrics.classification_hits.get(pr, 0) + 1
        return True

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        budget = max(1, self._settings.runtime.hard_rt_budget_ms) / 1000.0
        next_yield = loop.time() + budget

        while self._running:
            if not self._settings.runtime.use_priority_queue:
                msg = await self._src.get()
                await self._dst.put(msg)
                self._metrics.total_processed += 1
            else:
                get_task = asyncio.create_task(self._src.get())
                flush_task = asyncio.create_task(asyncio.sleep(0.01))

                done, _ = await asyncio.wait({get_task, flush_task}, return_when=asyncio.FIRST_COMPLETED)

                if flush_task not in done:
                    flush_task.cancel()

                if get_task in done:
                    msg = get_task.result()
                    self._enqueue(msg, loop.time())
                else:
                    get_task.cancel()

                await self._dispatch_one(loop)

            if loop.time() >= next_yield:
                await asyncio.sleep(0)
                next_yield = loop.time() + budget


def start_priority_dispatcher_task(src: asyncio.Queue[str], dst: asyncio.Queue[str], settings: Settings) -> asyncio.Task[None]:
    dispatcher = PriorityDispatcher(src, dst, settings)
    return asyncio.create_task(dispatcher.run(), name="priority-dispatcher-service")
