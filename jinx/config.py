from __future__ import annotations

import getpass
import os
import platform
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple


class Tag(str, Enum):
    PYTHON = "python"
    PYTHON_QUESTION = "python_question"
    MACHINE = "machine"


CODE_TAGS: FrozenSet[str] = frozenset({Tag.PYTHON, Tag.PYTHON_QUESTION})
ALL_TAGS: FrozenSet[str] = frozenset({Tag.MACHINE, *CODE_TAGS})


@dataclass(slots=True, frozen=True)
class HostInfo:
    os: str
    arch: str
    host: str
    user: str

    @classmethod
    def capture(cls) -> "HostInfo":
        return cls(
            os=f"{platform.system()} {platform.release()}",
            arch=platform.machine(),
            host=platform.node(),
            user=getpass.getuser(),
        )

    def to_dict(self) -> Dict[str, str]:
        return {"os": self.os, "arch": self.arch, "host": self.host, "user": self.user}


@dataclass(slots=True, frozen=True)
class TagBlock:
    start: str
    end: str


class PromptRegistry:
    __slots__ = ("_active",)

    def __init__(self, default: str = "burning_logic") -> None:
        self._active: Optional[str] = default

    def get(self) -> Optional[str]:
        return self._active

    def set(self, name: Optional[str]) -> None:
        self._active = (name or "").strip().lower() or None


_prompt_registry = PromptRegistry()
PROMPT_NAME: Optional[str] = _prompt_registry.get()


def set_prompt(name: Optional[str]) -> None:
    global PROMPT_NAME
    _prompt_registry.set(name)
    PROMPT_NAME = _prompt_registry.get()


def neon_stat() -> Dict[str, str]:
    return HostInfo.capture().to_dict()


def generate_fuse_id() -> str:
    return uuid.uuid4().hex[:12]


def jinx_tag() -> Tuple[str, Dict[str, TagBlock]]:
    fuse = generate_fuse_id()
    flames = {tag: TagBlock(start=f"<{tag}_{fuse}>\n", end=f"</{tag}_{fuse}>") for tag in ALL_TAGS}
    return fuse, flames
