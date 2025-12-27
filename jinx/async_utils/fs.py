from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import aiofiles
from aiofiles import ospath


@dataclass(slots=True)
class CacheEntry:
    content: str
    size: int
    timestamp: float


class FileCache:
    __slots__ = ("_cache", "_total_size", "_lock", "_capacity", "_ttl_s", "_max_bytes")

    def __init__(self, capacity: int = 256, ttl_s: float = 300, max_bytes: int = 10 * 1024 * 1024) -> None:
        self._cache: OrderedDict[Tuple[str, int, int], CacheEntry] = OrderedDict()
        self._total_size = 0
        self._lock = asyncio.Lock()
        self._capacity = capacity
        self._ttl_s = ttl_s
        self._max_bytes = max_bytes

    async def get(self, key: Tuple[str, int, int]) -> Optional[str]:
        async with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if (time.time() - entry.timestamp) >= self._ttl_s:
                self._total_size -= entry.size
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry.content

    async def put(self, key: Tuple[str, int, int], content: str) -> None:
        size = len(content.encode("utf-8", errors="ignore"))
        async with self._lock:
            while self._cache and (self._total_size + size) > self._max_bytes:
                _, old = self._cache.popitem(last=False)
                self._total_size -= old.size

            self._cache[key] = CacheEntry(content, size, time.time())
            self._total_size += size

            while len(self._cache) > self._capacity:
                _, old = self._cache.popitem(last=False)
                self._total_size -= old.size

    async def invalidate(self, path: str) -> None:
        async with self._lock:
            keys = [k for k in self._cache if k[0] == path]
            for k in keys:
                entry = self._cache.pop(k, None)
                if entry:
                    self._total_size -= entry.size


_file_cache = FileCache()


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


async def read_text_raw(path: str) -> str:
    if not path or not await ospath.exists(path) or not await ospath.isfile(path):
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return await f.read() or ""


async def read_text(path: str) -> str:
    return (await read_text_raw(path)).strip()


async def append_line(path: str, text: str) -> None:
    _ensure_dir(path)
    async with aiofiles.open(path, "a", encoding="utf-8") as f:
        await f.write(f"{text or ''}\n")


async def append_and_trim(path: str, text: str, keep_lines: int = 500) -> None:
    _ensure_dir(path)
    lines: list[str] = []
    if await ospath.exists(path):
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            lines = (await f.read()).splitlines()
    lines.extend(["", text])
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write("\n".join(lines[-keep_lines:]) + "\n")


async def write_text(path: str, text: str) -> None:
    _ensure_dir(path)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(text or "")
    await _file_cache.invalidate(path)


def _read_sync(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


async def read_text_abs_thread(path: str) -> str:
    st = os.stat(path)
    key = (path, int(st.st_mtime), int(st.st_size))

    cached = await _file_cache.get(key)
    if cached is not None:
        return cached

    content = await asyncio.to_thread(_read_sync, path)
    await _file_cache.put(key, content)
    return content
