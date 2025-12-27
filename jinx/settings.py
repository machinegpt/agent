from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from jinx.micro.common.config import clamp_float, clamp_int

T = TypeVar("T")


def _is_on(val: Optional[str]) -> bool:
    return (val or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    seen: set[str] = set()
    return [p for part in raw.split(",") if (p := part.strip()) and p not in seen and not seen.add(p)]


def _auto_threads() -> int:
    return max(4, min(32, (os.cpu_count() or 4) * 4))


def _auto_queue_maxsize() -> int:
    return max(100, min(2000, 100 + (os.cpu_count() or 1) * 50))


def _auto_rt_budget_ms() -> int:
    return 30 if (os.cpu_count() or 1) >= 8 else 40


@dataclass(slots=True)
class RuntimeSettings:
    queue_maxsize: int = 100
    use_priority_queue: bool = False
    queue_policy: str = "block"
    supervise_tasks: bool = True
    autorestart_limit: int = 5
    backoff_min_ms: int = 50
    backoff_max_ms: int = 2000
    hard_rt_budget_ms: int = 40
    threads_max_workers: int = 8
    auto_tune: bool = True
    saturate_enable_ratio: float = 0.6
    saturate_disable_ratio: float = 0.25
    saturate_window_ms: int = 500


@dataclass(slots=True)
class OpenAISettings:
    api_key: Optional[str] = None
    model: str = "gpt-5"
    proxy: Optional[str] = None
    vector_store_ids: List[str] = field(default_factory=list)
    force_file_search: bool = True


class SettingsBuilder:
    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> "SettingsBuilder":
        self._data[key] = value
        return self

    def pulse(self, v: int) -> "SettingsBuilder":
        return self.set("pulse", v)

    def timeout(self, v: int) -> "SettingsBuilder":
        return self.set("timeout", v)

    def model(self, v: str) -> "SettingsBuilder":
        return self.set("openai_model", v)

    def api_key(self, v: str) -> "SettingsBuilder":
        return self.set("openai_api_key", v)

    def build(self) -> "Settings":
        return Settings.from_env(self._data)


@dataclass(slots=True)
class Settings:
    pulse: int = 100
    timeout: int = 30
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @staticmethod
    def builder() -> SettingsBuilder:
        return SettingsBuilder()

    @staticmethod
    def from_env(overrides: Optional[Dict[str, Any]] = None) -> "Settings":
        o = overrides or {}

        s = Settings(
            pulse=clamp_int(int(o.get("pulse", 100)), 20, 1000),
            timeout=clamp_int(int(o.get("timeout", 30)), 5, 86400),
        )

        s.openai.api_key = str(o.get("openai_api_key", "")) or None
        s.openai.model = str(o.get("openai_model", s.openai.model))
        s.openai.proxy = str(o.get("proxy", "")) or None
        s.openai.vector_store_ids = _parse_csv(str(o.get("vector_store_ids", "")))
        s.openai.force_file_search = _is_on(str(o.get("force_file_search", "1")))

        rt = s.runtime
        rt.queue_maxsize = clamp_int(int(o.get("queue_maxsize", _auto_queue_maxsize())), 50, 10000)
        rt.use_priority_queue = _is_on(str(o.get("use_priority_queue", "1")))
        rt.queue_policy = str(o.get("queue_policy", rt.queue_policy))
        rt.supervise_tasks = not _is_on(str(o.get("no_supervisor", "0")))
        rt.autorestart_limit = clamp_int(int(o.get("autorestart_limit", rt.autorestart_limit)), 0, 50)
        rt.backoff_min_ms = clamp_int(int(o.get("backoff_min_ms", rt.backoff_min_ms)), 10, 10000)
        rt.backoff_max_ms = clamp_int(int(o.get("backoff_max_ms", rt.backoff_max_ms)), rt.backoff_min_ms, 60000)
        rt.hard_rt_budget_ms = clamp_int(int(o.get("hard_rt_budget_ms", _auto_rt_budget_ms())), 10, 200)
        rt.threads_max_workers = clamp_int(int(o.get("threads", _auto_threads())), 2, 64)
        rt.auto_tune = not _is_on(str(o.get("no_autotune", "0")))
        rt.saturate_enable_ratio = clamp_float(float(o.get("saturate_enable_ratio", rt.saturate_enable_ratio)), 0.0, 1.0)
        rt.saturate_disable_ratio = clamp_float(float(o.get("saturate_disable_ratio", rt.saturate_disable_ratio)), 0.0, 1.0)
        rt.saturate_window_ms = clamp_int(int(o.get("saturate_window_ms", rt.saturate_window_ms)), 50, 10000)

        return s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def apply_to_state(self) -> None:
        import jinx.state as jx_state
        jx_state.pulse = self.pulse
        jx_state.boom_limit = self.timeout
