"""
Compact RT-compatible code evolution core.

Integrates with existing Jinx Brain/AutoBrain systems.
All operations bounded, async-native, minimal memory footprint.
"""

from __future__ import annotations

import ast
import hashlib
import time
import asyncio
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass

_index_cache: dict[str, "CodeHash"] = {}
_max_cache = 500


class CodeHash(NamedTuple):
    struct: str
    token: str


@dataclass(slots=True)
class PatchResult:
    ok: bool
    msg: str
    dt_ms: float = 0.0


def get_index() -> dict[str, CodeHash]:
    return _index_cache


def compute_hash(code: str, max_depth: int = 30) -> CodeHash:
    try:
        tree = ast.parse(code)
        parts = []
        
        def visit(n: ast.AST, d: int = 0):
            if d > max_depth:
                return
            if isinstance(n, ast.Name):
                parts.append("V")
            elif isinstance(n, ast.Constant):
                parts.append("C")
            else:
                parts.append(type(n).__name__[:3])
            for c in ast.iter_child_nodes(n):
                visit(c, d + 1)
        
        visit(tree)
        struct = hashlib.md5("".join(parts).encode()).hexdigest()[:12]
    except SyntaxError:
        struct = "err"
    
    tokens = code.split()[:200]
    token = hashlib.md5(" ".join(tokens).encode()).hexdigest()[:12]
    
    return CodeHash(struct, token)


def check_syntax(code: str) -> tuple[bool, str | None]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"L{e.lineno}: {e.msg}"


def check_compat(old: str, new: str) -> tuple[bool, str | None]:
    def sigs(code: str) -> dict[str, int]:
        try:
            tree = ast.parse(code)
            out = {}
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    req = len(n.args.args) - len(n.args.defaults)
                    out[n.name] = req
            return out
        except SyntaxError:
            return {}
    
    old_s, new_s = sigs(old), sigs(new)
    for name, req in old_s.items():
        if name in new_s and new_s[name] > req:
            return False, f"{name}: +args"
    return True, None


def apply_patch(
    path: Path,
    new_code: str,
    backup: bool = True,
) -> PatchResult:
    t0 = time.perf_counter()
    
    ok, err = check_syntax(new_code)
    if not ok:
        return PatchResult(False, err or "syntax", (time.perf_counter() - t0) * 1000)
    
    path = Path(path)
    old_code = ""
    if path.exists():
        old_code = path.read_text(encoding="utf-8", errors="ignore")
        ok, err = check_compat(old_code, new_code)
        if not ok:
            return PatchResult(False, err or "compat", (time.perf_counter() - t0) * 1000)
    
    if backup and old_code:
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            bak.write_text(old_code, encoding="utf-8")
        except Exception:
            pass
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_code, encoding="utf-8")
        dt = (time.perf_counter() - t0) * 1000
        
        try:
            from jinx.micro.runtime.plugins import publish_event
            publish_event("evolve.patch", {"path": str(path), "dt_ms": dt})
        except Exception:
            pass
        
        return PatchResult(True, "ok", dt)
    except Exception as e:
        return PatchResult(False, str(e)[:80], (time.perf_counter() - t0) * 1000)


def record_change(
    path: str,
    reason: str,
    diff_lines: int = 0,
    ok: bool = True,
) -> None:
    try:
        from jinx.micro.brain import emit_learning_event
        emit_learning_event("code_change", {
            "path": path,
            "reason": reason[:100],
            "diff": diff_lines,
            "ok": ok,
            "ts": time.time(),
        })
    except Exception:
        pass
    
    try:
        from jinx.observability.metrics import record_patch_event
        record_patch_event("evolve", reason[:50], ok, diff_lines)
    except Exception:
        pass


async def eval_quality(code: str, timeout_ms: float = 50.0) -> dict:
    import asyncio
    t0 = time.perf_counter()
    result = {"ok": True, "score": 1.0, "issues": []}
    
    ok, err = check_syntax(code)
    if not ok:
        result["ok"] = False
        result["score"] = 0.0
        result["issues"].append(err)
        return result
    
    try:
        tree = ast.parse(code)
        cc = 1
        for n in ast.walk(tree):
            if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                cc += 1
        if cc > 20:
            result["score"] -= 0.2
            result["issues"].append(f"cc={cc}")
        
        lines = len(code.splitlines())
        if lines > 500:
            result["score"] -= 0.1
            result["issues"].append(f"lines={lines}")
    except Exception:
        pass
    
    result["dt_ms"] = (time.perf_counter() - t0) * 1000
    return result


def find_similar(code: str, index: dict[str, CodeHash] | None = None, threshold: int = 8) -> list[str]:
    if index is None:
        index = _index_cache
    h = compute_hash(code)
    similar = []
    for path, ch in index.items():
        match = sum(a == b for a, b in zip(h.struct, ch.struct))
        if match >= threshold:
            similar.append(path)
    return similar[:5]


def index_file(path: Path) -> CodeHash | None:
    try:
        code = Path(path).read_text(encoding="utf-8", errors="ignore")
        h = compute_hash(code)
        key = str(path)
        if len(_index_cache) >= _max_cache:
            try:
                _index_cache.pop(next(iter(_index_cache)))
            except Exception:
                pass
        _index_cache[key] = h
        return h
    except Exception:
        return None


async def apply_patch_async(
    path: Path,
    new_code: str,
    backup: bool = True,
) -> PatchResult:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, apply_patch, path, new_code, backup)


async def auto_heal(path: Path, error: str) -> PatchResult | None:
    try:
        from jinx.micro.brain import get_self_healer, heal_code
        code = Path(path).read_text(encoding="utf-8", errors="ignore")
        result = await heal_code(code, error)
        if result and result.healed_code:
            return await apply_patch_async(path, result.healed_code)
    except Exception:
        pass
    return None


def on_file_change(path: str) -> None:
    try:
        p = Path(path)
        if p.suffix == ".py" and p.exists():
            index_file(p)
    except Exception:
        pass


def init_index(root: Path, max_files: int = 200) -> int:
    count = 0
    try:
        for p in Path(root).rglob("*.py"):
            if count >= max_files:
                break
            if "__pycache__" in str(p):
                continue
            if index_file(p):
                count += 1
    except Exception:
        pass
    return count
