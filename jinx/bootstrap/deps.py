from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType
from typing import Callable, Dict, Iterable, Optional


@dataclass(slots=True)
class DependencyInfo:
    name: str
    module: Optional[ModuleType] = None
    load_time_ms: float = 0.0
    available: bool = False
    version: str = ""


class DependencyRegistry:
    __slots__ = ("_deps", "_load_hooks")

    def __init__(self) -> None:
        self._deps: Dict[str, DependencyInfo] = {}
        self._load_hooks: list[Callable[[DependencyInfo], None]] = []

    def register_hook(self, hook: Callable[[DependencyInfo], None]) -> None:
        self._load_hooks.append(hook)

    def load(self, name: str) -> Optional[ModuleType]:
        if name in self._deps and self._deps[name].available:
            return self._deps[name].module

        info = DependencyInfo(name=name)
        t0 = time.perf_counter()

        mod = import_module(name)
        info.module = mod
        info.available = True
        info.load_time_ms = (time.perf_counter() - t0) * 1000
        info.version = getattr(mod, "__version__", "")

        self._deps[name] = info
        for hook in self._load_hooks:
            hook(info)

        return mod

    def get(self, name: str) -> Optional[DependencyInfo]:
        return self._deps.get(name)

    def loaded(self) -> Dict[str, DependencyInfo]:
        return {k: v for k, v in self._deps.items() if v.available}

    def stats(self) -> Dict[str, float]:
        return {name: info.load_time_ms for name, info in self._deps.items() if info.available}


_registry = DependencyRegistry()


def ensure_optional(packages: Iterable[str]) -> Dict[str, ModuleType]:
    mods: Dict[str, ModuleType] = {}
    for name in packages:
        mod = _registry.load(name)
        if mod:
            mods[name] = mod
    return mods


def get_dependency(name: str) -> Optional[ModuleType]:
    return _registry.load(name)


def get_registry() -> DependencyRegistry:
    return _registry
