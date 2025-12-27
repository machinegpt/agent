from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Optional, Set, Tuple

_REPAIR_STATE_DIR: Path = Path(".jinx/repair")
_REPAIR_HISTORY: Path = Path(".jinx/repair/history.json")
_FAILURE_LOG: Path = Path(".jinx/repair/failures.json")
_MAX_REPAIR_ITERATIONS: int = 5
_MAX_REPAIR_ATTEMPTS: int = 3
_MAX_FILES_PER_REPAIR: int = 10
_CONTEXT_LINES: int = 50
_CIRCUIT_BREAKER_THRESHOLD: int = 3
_CIRCUIT_BREAKER_WINDOW_S: float = 300.0
_BACKOFF_BASE_S: float = 2.0
_BACKOFF_MAX_S: float = 60.0


class FailureTracker:
    __slots__ = ("_failures", "_blocked", "_path")

    def __init__(self) -> None:
        self._failures: Dict[str, List[float]] = {}
        self._blocked: Dict[str, float] = {}
        self._path = _FAILURE_LOG
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._failures = data.get("failures", {})
                self._blocked = data.get("blocked", {})
            except Exception:
                pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "failures": self._failures,
            "blocked": self._blocked,
        }))

    def _pattern_key(self, error_msg: str, file_path: str) -> str:
        msg_hash = hashlib.md5(error_msg[:100].encode()).hexdigest()[:8]
        return f"{Path(file_path).name}:{msg_hash}"

    def record_failure(self, error_msg: str, file_path: str, code_hash: str = "") -> None:
        now = time.time()
        key = self._pattern_key(error_msg, file_path)
        
        if key not in self._failures:
            self._failures[key] = []
        self._failures[key].append(now)
        
        recent = [t for t in self._failures[key] if now - t < _CIRCUIT_BREAKER_WINDOW_S]
        self._failures[key] = recent[-20:]
        
        if len(recent) >= _CIRCUIT_BREAKER_THRESHOLD:
            backoff = min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * (2 ** (len(recent) - _CIRCUIT_BREAKER_THRESHOLD)))
            self._blocked[key] = now + backoff
        
        self._save()
        self._notify_brain(error_msg, file_path, len(recent))

    def is_blocked(self, error_msg: str, file_path: str) -> Tuple[bool, float]:
        now = time.time()
        key = self._pattern_key(error_msg, file_path)
        
        if key in self._blocked:
            unblock_at = self._blocked[key]
            if now < unblock_at:
                return True, unblock_at - now
            del self._blocked[key]
        return False, 0.0

    def record_success(self, error_msg: str, file_path: str) -> None:
        key = self._pattern_key(error_msg, file_path)
        if key in self._failures:
            self._failures[key] = self._failures[key][-2:]
        if key in self._blocked:
            del self._blocked[key]
        self._save()

    def get_failure_count(self, error_msg: str, file_path: str) -> int:
        key = self._pattern_key(error_msg, file_path)
        now = time.time()
        recent = [t for t in self._failures.get(key, []) if now - t < _CIRCUIT_BREAKER_WINDOW_S]
        return len(recent)

    def _notify_brain(self, error_msg: str, file_path: str, count: int) -> None:
        try:
            from jinx.micro.brain import learn_from_error
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(learn_from_error("", type(Exception).__name__, error_msg[:200]))
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        active_blocks = {k: v - now for k, v in self._blocked.items() if v > now}
        return {
            "tracked_patterns": len(self._failures),
            "active_blocks": len(active_blocks),
            "blocks": active_blocks,
        }


_failure_tracker: Optional[FailureTracker] = None


def get_failure_tracker() -> FailureTracker:
    global _failure_tracker
    if _failure_tracker is None:
        _failure_tracker = FailureTracker()
    return _failure_tracker


def should_attempt_repair(error_msg: str, file_path: str) -> Tuple[bool, str]:
    tracker = get_failure_tracker()
    blocked, wait_s = tracker.is_blocked(error_msg, file_path)
    if blocked:
        return False, f"Circuit breaker active, wait {wait_s:.0f}s"
    
    count = tracker.get_failure_count(error_msg, file_path)
    if count >= _CIRCUIT_BREAKER_THRESHOLD:
        return False, f"Too many failures ({count}), backing off"
    
    return True, ""


class ErrorCategory(IntEnum):
    UNKNOWN = 0
    SYNTAX_ERROR = 1
    IMPORT_ERROR = 2
    ATTRIBUTE_ERROR = 3
    TYPE_ERROR = 4
    NAME_ERROR = 5
    RUNTIME_ERROR = 6
    VALUE_ERROR = 7


class RepairStrategy(IntEnum):
    REMOVE_ASSIGNMENT = 0
    REMOVE_SLOTS_CONFLICT = 1
    COMMENT_OUT = 2
    ADD_IMPORT = 3
    FIX_TYPE_HINT = 4
    WRAP_TRY_EXCEPT = 5
    REPLACE_CODE = 6
    ADD_ATTRIBUTE = 7


class RepairPhase(IntEnum):
    ANALYZE = 0
    LOCATE = 1
    PLAN = 2
    EXECUTE = 3
    VERIFY = 4
    LEARN = 5


@dataclass(slots=True)
class ErrorSignature:
    category: ErrorCategory
    message: str
    file_path: str
    line_number: int
    code_context: str
    attribute: Optional[str] = None
    class_name: Optional[str] = None
    pattern_hash: int = 0

    def __post_init__(self) -> None:
        self.pattern_hash = hash((self.category, self.message[:50], self.attribute))


@dataclass(slots=True)
class RepairAction:
    strategy: RepairStrategy
    file_path: str
    line_number: int
    old_code: str
    new_code: str
    confidence: float = 0.0
    description: str = ""


@dataclass(slots=True)
class RepairResult:
    success: bool
    action: Optional[RepairAction] = None
    error: Optional[str] = None
    attempts: int = 0


class ErrorAnalyzer:
    __slots__ = ("_patterns", "_cache")

    def __init__(self) -> None:
        self._patterns: List[Tuple[re.Pattern, ErrorCategory, Callable]] = []
        self._cache: Dict[int, ErrorSignature] = {}
        self._register_patterns()

    def _register_patterns(self) -> None:
        self._patterns = [
            (
                re.compile(r"'(\w+)' object attribute '(\w+)' is read-only"),
                ErrorCategory.ATTRIBUTE_ERROR,
                self._extract_readonly_attr,
            ),
            (
                re.compile(r"cannot set attribute '?(\w+)'?"),
                ErrorCategory.ATTRIBUTE_ERROR,
                self._extract_cannot_set,
            ),
            (
                re.compile(r"'(\w+)' object has no attribute '(\w+)'"),
                ErrorCategory.ATTRIBUTE_ERROR,
                self._extract_no_attr,
            ),
            (
                re.compile(r"No module named '(\w+)'"),
                ErrorCategory.IMPORT_ERROR,
                self._extract_missing_module,
            ),
            (
                re.compile(r"cannot import name '(\w+)' from '(\w+)'"),
                ErrorCategory.IMPORT_ERROR,
                self._extract_import_name,
            ),
            (
                re.compile(r"name '(\w+)' is not defined"),
                ErrorCategory.NAME_ERROR,
                self._extract_undefined_name,
            ),
            (
                re.compile(r"expected '(\w+)' but got '(\w+)'"),
                ErrorCategory.TYPE_ERROR,
                self._extract_type_mismatch,
            ),
        ]

    def _extract_readonly_attr(self, match: re.Match, tb_info: Dict) -> Dict:
        return {"class_name": match.group(1), "attribute": match.group(2)}

    def _extract_cannot_set(self, match: re.Match, tb_info: Dict) -> Dict:
        return {"attribute": match.group(1)}

    def _extract_no_attr(self, match: re.Match, tb_info: Dict) -> Dict:
        return {"class_name": match.group(1), "attribute": match.group(2)}

    def _extract_missing_module(self, match: re.Match, tb_info: Dict) -> Dict:
        return {"module": match.group(1)}

    def _extract_import_name(self, match: re.Match, tb_info: Dict) -> Dict:
        return {"name": match.group(1), "module": match.group(2)}

    def _extract_undefined_name(self, match: re.Match, tb_info: Dict) -> Dict:
        return {"name": match.group(1)}

    def _extract_type_mismatch(self, match: re.Match, tb_info: Dict) -> Dict:
        return {"expected": match.group(1), "got": match.group(2)}

    def _parse_traceback(self, tb_str: str) -> List[Dict]:
        frames = []
        pattern = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
        for match in pattern.finditer(tb_str):
            frames.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "function": match.group(3),
            })
        return frames

    def _get_code_context(self, file_path: str, line_number: int, context: int = 3) -> str:
        try:
            path = Path(file_path)
            if not path.exists():
                return ""
            lines = path.read_text(encoding="utf-8").splitlines()
            start = max(0, line_number - context - 1)
            end = min(len(lines), line_number + context)
            return "\n".join(lines[start:end])
        except Exception:
            return ""

    def analyze(self, exception: BaseException, tb_str: str = "") -> ErrorSignature:
        if not tb_str:
            tb_str = traceback.format_exception(type(exception), exception, exception.__traceback__)
            tb_str = "".join(tb_str)

        msg = str(exception)
        category = ErrorCategory.UNKNOWN
        extra: Dict[str, Any] = {}

        for pattern, cat, extractor in self._patterns:
            match = pattern.search(msg)
            if match:
                category = cat
                extra = extractor(match, {"tb": tb_str})
                break

        if category == ErrorCategory.UNKNOWN:
            if isinstance(exception, AttributeError):
                category = ErrorCategory.ATTRIBUTE_ERROR
            elif isinstance(exception, ImportError):
                category = ErrorCategory.IMPORT_ERROR
            elif isinstance(exception, TypeError):
                category = ErrorCategory.TYPE_ERROR
            elif isinstance(exception, NameError):
                category = ErrorCategory.NAME_ERROR
            elif isinstance(exception, SyntaxError):
                category = ErrorCategory.SYNTAX_ERROR

        frames = self._parse_traceback(tb_str)
        file_path = ""
        line_number = 0
        if frames:
            for frame in reversed(frames):
                if not any(x in frame["file"] for x in ["<", "site-packages", "lib/python"]):
                    file_path = frame["file"]
                    line_number = frame["line"]
                    break

        code_context = self._get_code_context(file_path, line_number) if file_path else ""

        return ErrorSignature(
            category=category,
            message=msg,
            file_path=file_path,
            line_number=line_number,
            code_context=code_context,
            attribute=extra.get("attribute"),
            class_name=extra.get("class_name"),
        )


class RepairGenerator:
    __slots__ = ("_strategies", "_history", "_known_fixes")

    def __init__(self) -> None:
        self._strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.ATTRIBUTE_ERROR: [
                self._repair_readonly_attribute,
                self._repair_slots_assignment,
                self._repair_monkey_patch_removal,
            ],
            ErrorCategory.IMPORT_ERROR: [
                self._repair_missing_import,
                self._repair_circular_import,
            ],
            ErrorCategory.TYPE_ERROR: [
                self._repair_type_annotation,
                self._repair_type_hint_any,
            ],
            ErrorCategory.NAME_ERROR: [
                self._repair_undefined_name,
                self._repair_missing_global,
            ],
            ErrorCategory.SYNTAX_ERROR: [
                self._repair_syntax_error,
            ],
            ErrorCategory.RUNTIME_ERROR: [
                self._repair_runtime_guard,
            ],
        }
        self._history: List[RepairAction] = []
        self._known_fixes: Dict[str, Tuple[str, str]] = {
            "read-only": ("REMOVE_ASSIGNMENT", "Remove monkey-patch to __slots__ class"),
            "has no attribute": ("ADD_ATTRIBUTE", "Add missing attribute or use getattr"),
            "is not defined": ("ADD_IMPORT", "Add missing import or definition"),
            "cannot import": ("FIX_IMPORT", "Fix import path or add __init__.py"),
        }

    def _read_file(self, path: str) -> List[str]:
        return Path(path).read_text(encoding="utf-8").splitlines()

    def _find_assignment_line(self, lines: List[str], attr: str, start_line: int) -> Optional[int]:
        pattern = re.compile(rf"\.{attr}\s*=")
        for i in range(max(0, start_line - 10), min(len(lines), start_line + 10)):
            if pattern.search(lines[i]):
                return i
        return None

    def _find_class_with_slots(self, lines: List[str], class_name: str) -> Optional[Tuple[int, int]]:
        class_start = None
        for i, line in enumerate(lines):
            if f"class {class_name}" in line:
                class_start = i
            elif class_start is not None and "__slots__" in line:
                return (class_start, i)
        return None

    def _repair_readonly_attribute(self, sig: ErrorSignature) -> Optional[RepairAction]:
        if not sig.attribute or not sig.file_path:
            return None

        lines = self._read_file(sig.file_path)
        assign_line = self._find_assignment_line(lines, sig.attribute, sig.line_number)

        if assign_line is None:
            return None

        old_code = lines[assign_line]
        
        if f".{sig.attribute} = " in old_code:
            indent = len(old_code) - len(old_code.lstrip())
            new_code = " " * indent + f"pass  # REMOVED: {old_code.strip()}"
            
            return RepairAction(
                strategy=RepairStrategy.REMOVE_ASSIGNMENT,
                file_path=sig.file_path,
                line_number=assign_line + 1,
                old_code=old_code,
                new_code=new_code,
                confidence=0.95,
                description=f"Remove assignment to read-only attribute '{sig.attribute}'",
            )

        return None

    def _repair_monkey_patch_removal(self, sig: ErrorSignature) -> Optional[RepairAction]:
        if not sig.file_path or "read-only" not in sig.message:
            return None

        lines = self._read_file(sig.file_path)
        
        for i in range(max(0, sig.line_number - 5), min(len(lines), sig.line_number + 5)):
            line = lines[i]
            if re.search(r'\.\w+\s*=\s*\w+', line) and not line.strip().startswith("#"):
                if "self." not in line or "= self." in line:
                    indent = len(line) - len(line.lstrip())
                    new_code = " " * indent + f"pass  # DISABLED: {line.strip()}"
                    return RepairAction(
                        strategy=RepairStrategy.REMOVE_ASSIGNMENT,
                        file_path=sig.file_path,
                        line_number=i + 1,
                        old_code=line,
                        new_code=new_code,
                        confidence=0.85,
                        description="Remove monkey-patch assignment to __slots__ object",
                    )
        return None

    def _repair_slots_assignment(self, sig: ErrorSignature) -> Optional[RepairAction]:
        if not sig.file_path:
            return None

        lines = self._read_file(sig.file_path)
        
        for i, line in enumerate(lines):
            if "__slots__" in line and sig.class_name and sig.class_name in "".join(lines[max(0,i-20):i]):
                slots_match = re.search(r'__slots__\s*=\s*\(([^)]+)\)', line)
                if slots_match and sig.attribute:
                    current_slots = slots_match.group(1)
                    if f'"{sig.attribute}"' not in current_slots and f"'{sig.attribute}'" not in current_slots:
                        new_slots = current_slots.rstrip() + f', "{sig.attribute}"'
                        new_line = line.replace(current_slots, new_slots)
                        return RepairAction(
                            strategy=RepairStrategy.REMOVE_SLOTS_CONFLICT,
                            file_path=sig.file_path,
                            line_number=i + 1,
                            old_code=line,
                            new_code=new_line,
                            confidence=0.7,
                            description=f"Add '{sig.attribute}' to __slots__",
                        )
        return None

    def _repair_missing_import(self, sig: ErrorSignature) -> Optional[RepairAction]:
        if not sig.file_path:
            return None
        
        match = re.search(r"No module named '(\w+)'", sig.message)
        if not match:
            match = re.search(r"cannot import name '(\w+)'", sig.message)
        if not match:
            return None
        
        module_name = match.group(1)
        lines = self._read_file(sig.file_path)
        
        for i, line in enumerate(lines):
            if f"import {module_name}" in line or f"from {module_name}" in line:
                indent = len(line) - len(line.lstrip())
                new_code = " " * indent + f"# {line.strip()}  # DISABLED: module not found"
                return RepairAction(
                    strategy=RepairStrategy.COMMENT_OUT,
                    file_path=sig.file_path,
                    line_number=i + 1,
                    old_code=line,
                    new_code=new_code,
                    confidence=0.6,
                    description=f"Comment out missing import '{module_name}'",
                )
        return None

    def _repair_circular_import(self, sig: ErrorSignature) -> Optional[RepairAction]:
        if "circular" not in sig.message.lower():
            return None
        return None

    def _repair_type_annotation(self, sig: ErrorSignature) -> Optional[RepairAction]:
        return None

    def _repair_type_hint_any(self, sig: ErrorSignature) -> Optional[RepairAction]:
        if not sig.file_path:
            return None
        
        lines = self._read_file(sig.file_path)
        line_idx = sig.line_number - 1
        
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            if ": asyncio.Event" in line:
                new_line = line.replace(": asyncio.Event", ": Any")
                return RepairAction(
                    strategy=RepairStrategy.FIX_TYPE_HINT,
                    file_path=sig.file_path,
                    line_number=sig.line_number,
                    old_code=line,
                    new_code=new_line,
                    confidence=0.8,
                    description="Change type hint to Any for compatibility",
                )
        return None

    def _repair_undefined_name(self, sig: ErrorSignature) -> Optional[RepairAction]:
        match = re.search(r"name '(\w+)' is not defined", sig.message)
        if not match or not sig.file_path:
            return None
        
        name = match.group(1)
        lines = self._read_file(sig.file_path)
        line_idx = sig.line_number - 1
        
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            indent = len(line) - len(line.lstrip())
            
            if name in line and not line.strip().startswith("#"):
                new_code = " " * indent + f"pass  # DISABLED: {line.strip()}  # undefined: {name}"
                return RepairAction(
                    strategy=RepairStrategy.COMMENT_OUT,
                    file_path=sig.file_path,
                    line_number=sig.line_number,
                    old_code=line,
                    new_code=new_code,
                    confidence=0.5,
                    description=f"Comment out line with undefined name '{name}'",
                )
        return None

    def _repair_missing_global(self, sig: ErrorSignature) -> Optional[RepairAction]:
        return None

    def _repair_syntax_error(self, sig: ErrorSignature) -> Optional[RepairAction]:
        return None

    def _repair_runtime_guard(self, sig: ErrorSignature) -> Optional[RepairAction]:
        if not sig.file_path:
            return None
        
        lines = self._read_file(sig.file_path)
        line_idx = sig.line_number - 1
        
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            indent = len(line) - len(line.lstrip())
            
            new_lines = [
                " " * indent + "try:",
                " " * (indent + 4) + line.strip(),
                " " * indent + "except Exception:",
                " " * (indent + 4) + "pass",
            ]
            
            return RepairAction(
                strategy=RepairStrategy.WRAP_TRY_EXCEPT,
                file_path=sig.file_path,
                line_number=sig.line_number,
                old_code=line,
                new_code="\n".join(new_lines),
                confidence=0.4,
                description="Wrap problematic code in try/except",
            )
        return None

    def generate(self, sig: ErrorSignature) -> List[RepairAction]:
        actions = []
        strategies = self._strategies.get(sig.category, [])
        
        for strategy_fn in strategies:
            action = strategy_fn(sig)
            if action:
                actions.append(action)

        for keyword, (strategy_name, desc) in self._known_fixes.items():
            if keyword in sig.message.lower() and not actions:
                pass

        actions.sort(key=lambda a: -a.confidence)
        return actions


class CodePatcher:
    __slots__ = ("_backup_dir",)

    def __init__(self) -> None:
        self._backup_dir = Path(".jinx/repair_backups")
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def _backup(self, file_path: str) -> Path:
        path = Path(file_path)
        backup = self._backup_dir / f"{path.name}.{int(time.time())}.bak"
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup

    def apply(self, action: RepairAction) -> bool:
        path = Path(action.file_path)
        if not path.exists():
            return False

        self._backup(action.file_path)

        lines = path.read_text(encoding="utf-8").splitlines()
        line_idx = action.line_number - 1

        if 0 <= line_idx < len(lines):
            if lines[line_idx].strip() == action.old_code.strip():
                lines[line_idx] = action.new_code
            else:
                for i, line in enumerate(lines):
                    if line.strip() == action.old_code.strip():
                        lines[i] = action.new_code
                        break

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True

    def revert(self, file_path: str) -> bool:
        path = Path(file_path)
        backups = sorted(self._backup_dir.glob(f"{path.name}.*.bak"), reverse=True)
        if backups:
            path.write_text(backups[0].read_text(encoding="utf-8"), encoding="utf-8")
            return True
        return False


class RepairHistory:
    __slots__ = ("_path", "_repairs")

    def __init__(self) -> None:
        self._path = _REPAIR_HISTORY
        self._repairs: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            self._repairs = json.loads(self._path.read_text())

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._repairs, indent=2))

    def record(self, sig: ErrorSignature, action: RepairAction, success: bool) -> None:
        self._repairs.append({
            "timestamp": time.time(),
            "pattern_hash": sig.pattern_hash,
            "category": int(sig.category),
            "message": sig.message[:200],
            "file": action.file_path,
            "line": action.line_number,
            "strategy": int(action.strategy),
            "success": success,
        })
        self._save()

    def get_success_rate(self, pattern_hash: int) -> float:
        relevant = [r for r in self._repairs if r["pattern_hash"] == pattern_hash]
        if not relevant:
            return 0.5
        return sum(1 for r in relevant if r["success"]) / len(relevant)


class AIRepairEngine:
    __slots__ = ("_api_key", "_model", "_enabled")

    def __init__(self) -> None:
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._enabled = bool(self._api_key)

    def _build_prompt(self, sig: ErrorSignature, file_content: str) -> str:
        return f"""You are a Python code repair expert. Analyze this error and provide a fix.

ERROR TYPE: {sig.category.name}
ERROR MESSAGE: {sig.message}
FILE: {sig.file_path}
LINE: {sig.line_number}

CODE CONTEXT:
```python
{sig.code_context}
```

FULL FILE:
```python
{file_content[:8000]}
```

Provide ONLY the fixed code for the problematic section. Output format:
<analysis>Brief explanation of the bug</analysis>
<fix>
The exact fixed code (only the lines that need to change)
</fix>
<old_code>The exact original code that should be replaced</old_code>
<confidence>0.0-1.0</confidence>"""

    def repair_sync(self, sig: ErrorSignature) -> Optional[RepairAction]:
        if not self._enabled or not sig.file_path:
            return None

        try:
            import urllib.request
            import json as _json

            file_content = Path(sig.file_path).read_text(encoding="utf-8")
            prompt = self._build_prompt(sig, file_content)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            data = _json.dumps({
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.2,
            }).encode()

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=data,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = _json.loads(resp.read().decode())

            response = result["choices"][0]["message"]["content"]
            return self._parse_response(response, sig)

        except Exception:
            return None

    def _parse_response(self, response: str, sig: ErrorSignature) -> Optional[RepairAction]:
        import re

        analysis_match = re.search(r"<analysis>(.*?)</analysis>", response, re.DOTALL)
        fix_match = re.search(r"<fix>(.*?)</fix>", response, re.DOTALL)
        old_match = re.search(r"<old_code>(.*?)</old_code>", response, re.DOTALL)
        conf_match = re.search(r"<confidence>([\d.]+)</confidence>", response)

        if not fix_match or not old_match:
            return None

        new_code = fix_match.group(1).strip()
        old_code = old_match.group(1).strip()
        confidence = float(conf_match.group(1)) if conf_match else 0.7
        description = analysis_match.group(1).strip() if analysis_match else "AI-generated fix"

        if not new_code or not old_code:
            return None

        return RepairAction(
            strategy=RepairStrategy.REPLACE_CODE,
            file_path=sig.file_path,
            line_number=sig.line_number,
            old_code=old_code,
            new_code=new_code,
            confidence=confidence,
            description=f"[AI] {description}",
        )


class SelfRepairAgent:
    __slots__ = ("_analyzer", "_generator", "_patcher", "_history", "_max_attempts", "_ai_engine")

    def __init__(self) -> None:
        self._analyzer = ErrorAnalyzer()
        self._generator = RepairGenerator()
        self._patcher = CodePatcher()
        self._history = RepairHistory()
        self._max_attempts = _MAX_REPAIR_ATTEMPTS
        self._ai_engine = AIRepairEngine()

    def diagnose(self, exception: BaseException, tb_str: str = "") -> ErrorSignature:
        return self._analyzer.analyze(exception, tb_str)

    def repair(self, sig: ErrorSignature) -> RepairResult:
        actions = self._generator.generate(sig)

        if self._ai_engine._enabled:
            ai_action = self._ai_engine.repair_sync(sig)
            if ai_action:
                actions.insert(0, ai_action)

        if not actions:
            return RepairResult(success=False, error="No repair strategy found")

        for action in actions:
            success_rate = self._history.get_success_rate(sig.pattern_hash)
            if "[AI]" in action.description:
                action.confidence = min(0.95, action.confidence * 1.2)
            else:
                action.confidence *= (0.5 + 0.5 * success_rate)

        actions.sort(key=lambda a: -a.confidence)
        best_action = actions[0]

        try:
            success = self._patcher.apply(best_action)
            self._history.record(sig, best_action, success)
            _notify_evolve(sig, best_action, success)
            return RepairResult(success=success, action=best_action)
        except Exception as e:
            _notify_evolve(sig, best_action, False)
            return RepairResult(success=False, action=best_action, error=str(e))

    def attempt_recovery(self, exception: BaseException, tb_str: str = "") -> RepairResult:
        sig = self.diagnose(exception, tb_str)
        return self.repair(sig)


def _notify_evolve(sig: ErrorSignature, action: RepairAction, success: bool) -> None:
    try:
        from jinx.evolve import record_change
        diff = len(action.new_code.splitlines()) - len(action.old_code.splitlines())
        record_change(action.file_path, f"repair:{sig.category.name}", abs(diff), success)
    except Exception:
        pass


_agent: Optional[SelfRepairAgent] = None


def get_repair_agent() -> SelfRepairAgent:
    global _agent
    if _agent is None:
        _agent = SelfRepairAgent()
    return _agent


def auto_repair_and_restart(exception: BaseException, restart_cmd: Optional[List[str]] = None) -> bool:
    agent = get_repair_agent()
    tracker = get_failure_tracker()
    tb_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    error_msg = str(exception)
    
    print(f"\n{'='*60}")
    print("🔧 JINX SELF-REPAIR SYSTEM ACTIVATED")
    print(f"{'='*60}")
    print(f"Error: {exception}")
    
    sig = agent.diagnose(exception, tb_str)
    file_path = sig.file_path or "unknown"
    
    can_repair, block_reason = should_attempt_repair(error_msg, file_path)
    if not can_repair:
        print(f"\n⏸️ Repair skipped: {block_reason}")
        print("   System is learning from repeated failures to avoid loops")
        tracker.record_failure(error_msg, file_path)
        return False
    
    failure_count = tracker.get_failure_count(error_msg, file_path)
    if failure_count > 0:
        print(f"\n⚠️ Previous failures for this pattern: {failure_count}")
    
    print(f"\nDiagnosis:")
    print(f"  Category: {sig.category.name}")
    print(f"  File: {file_path}:{sig.line_number}")
    if sig.attribute:
        print(f"  Attribute: {sig.attribute}")
    if sig.class_name:
        print(f"  Class: {sig.class_name}")
    
    ai_available = agent._ai_engine._enabled
    print(f"\n🤖 AI Repair: {'ENABLED' if ai_available else 'DISABLED (no OPENAI_API_KEY)'}")
    
    if ai_available:
        print("  Querying OpenAI for intelligent fix...")
    
    result = agent.repair(sig)
    
    if result.success and result.action:
        is_ai = "[AI]" in result.action.description
        print(f"\n✅ Repair applied {'(AI-powered)' if is_ai else '(pattern-based)'}:")
        print(f"  Strategy: {result.action.strategy.name}")
        print(f"  Description: {result.action.description}")
        print(f"  Confidence: {result.action.confidence:.0%}")
        
        tracker.record_success(error_msg, file_path)
        
        if restart_cmd:
            print(f"\n🔄 Restarting Jinx...")
            time.sleep(0.5)
            os.execv(sys.executable, [sys.executable] + restart_cmd)
        return True
    else:
        print(f"\n❌ Repair failed: {result.error}")
        tracker.record_failure(error_msg, file_path)
        
        stats = tracker.get_stats()
        if stats["active_blocks"]:
            print(f"\n📊 Circuit breaker status: {stats['active_blocks']} patterns blocked")
        
        if not ai_available:
            print("\n💡 Tip: Set OPENAI_API_KEY for AI-powered repairs")
        
        return False


async def delegate_to_runtime_healing(exception: BaseException, tb_str: str) -> bool:
    try:
        from jinx.micro.brain import heal_code
        frames = _parse_tb_frames(tb_str)
        if not frames:
            return False
        
        for frame in reversed(frames):
            if "__pycache__" in frame["file"] or "site-packages" in frame["file"]:
                continue
            path = Path(frame["file"])
            if path.exists():
                code = path.read_text(encoding="utf-8", errors="ignore")
                result = await heal_code(code, str(exception))
                if result and result.healed_code:
                    from jinx.evolve import apply_patch_async
                    patch_result = await apply_patch_async(path, result.healed_code)
                    if patch_result.ok:
                        return True
        return False
    except Exception:
        return False


def _parse_tb_frames(tb_str: str) -> List[Dict]:
    frames = []
    pattern = re.compile(r'File "([^"]+)", line (\d+)')
    for match in pattern.finditer(tb_str):
        frames.append({"file": match.group(1), "line": int(match.group(2))})
    return frames


def wrap_main_with_repair(main_fn: Callable, restart_args: Optional[List[str]] = None) -> Callable:
    def wrapped() -> Any:
        attempts = 0
        while attempts < _MAX_REPAIR_ATTEMPTS:
            try:
                return main_fn()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                attempts += 1
                print(f"\n[Attempt {attempts}/{_MAX_REPAIR_ATTEMPTS}]")
                if not auto_repair_and_restart(e, restart_args if attempts < _MAX_REPAIR_ATTEMPTS else None):
                    if attempts >= _MAX_REPAIR_ATTEMPTS:
                        raise
        return None
    return wrapped


def track_exec_error(error_msg: str, code: str, file_hint: str = "generated") -> None:
    tracker = get_failure_tracker()
    code_hash = hashlib.md5(code[:500].encode()).hexdigest()[:8]
    tracker.record_failure(error_msg, file_hint, code_hash)
    
    try:
        from jinx.micro.brain import learn_from_error
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(learn_from_error(code[:2000], "ExecError", error_msg[:200]))
        except RuntimeError:
            pass
    except Exception:
        pass


def can_retry_exec(error_msg: str, file_hint: str = "generated") -> bool:
    can, _ = should_attempt_repair(error_msg, file_hint)
    return can


__all__ = [
    "ErrorCategory",
    "RepairStrategy", 
    "ErrorSignature",
    "RepairAction",
    "RepairResult",
    "FailureTracker",
    "SelfRepairAgent",
    "AIRepairEngine",
    "get_repair_agent",
    "get_failure_tracker",
    "should_attempt_repair",
    "track_exec_error",
    "can_retry_exec",
    "auto_repair_and_restart",
    "wrap_main_with_repair",
    "delegate_to_runtime_healing",
]
