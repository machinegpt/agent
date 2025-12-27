from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(slots=True)
class EnvLoadResult:
    path: str
    variables_loaded: int
    variables_skipped: int
    loaded_keys: list[str] = field(default_factory=list)


class EnvLoader:
    __slots__ = ("_loaded", "_results")

    def __init__(self) -> None:
        self._loaded: Dict[str, str] = {}
        self._results: list[EnvLoadResult] = []

    @staticmethod
    def _strip_quotes(v: str) -> str:
        t = v.strip()
        if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
            return t[1:-1]
        return t

    @staticmethod
    def _default_paths() -> list[Path]:
        paths = []
        cwd = Path.cwd()
        paths.append(cwd / ".env")
        repo_root = Path(__file__).parent.parent.parent
        paths.append(repo_root / ".env")
        return [p for p in paths if p.exists()]

    def load_file(self, path: Path, override: bool = False) -> EnvLoadResult:
        result = EnvLoadResult(path=str(path), variables_loaded=0, variables_skipped=0)

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue

                value = self._strip_quotes(value)

                if not override and os.environ.get(key):
                    result.variables_skipped += 1
                    continue

                os.environ[key] = value
                self._loaded[key] = value
                result.variables_loaded += 1
                result.loaded_keys.append(key)

        self._results.append(result)
        return result

    def load(self, paths: Optional[Iterable[str]] = None, override: bool = False) -> list[EnvLoadResult]:
        target_paths = [Path(p) for p in paths] if paths else self._default_paths()
        return [self.load_file(p, override) for p in target_paths if p.exists()]

    def get_loaded(self) -> Dict[str, str]:
        return dict(self._loaded)

    def get_results(self) -> list[EnvLoadResult]:
        return list(self._results)


_loader = EnvLoader()


def load_env(paths: Optional[Iterable[str]] = None) -> list[EnvLoadResult]:
    return _loader.load(paths)


def get_env_loader() -> EnvLoader:
    return _loader
