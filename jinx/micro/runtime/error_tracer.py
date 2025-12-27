"""
Error Tracer - Collects and analyzes data to trace error sources.

Uses existing data from:
- crash_diagnostics: operation traces, stack traces
- events: recorded events from orchestrator
- logs: execution logs, error logs
- .jinx/: persistent state, metrics, learnings

Tracks execution flow from startup to error to identify root causes.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_TRACE_DIR = Path(".jinx/error_traces")
_MAX_TRACE_DEPTH = 50
_MAX_CODE_CONTEXT = 30


@dataclass(slots=True)
class CodeLocation:
    file: str
    line: int
    function: str
    code_snippet: str = ""


@dataclass(slots=True)
class ExecutionEvent:
    timestamp: float
    event_type: str
    details: Dict[str, Any]
    location: Optional[CodeLocation] = None


@dataclass(slots=True)
class ErrorTrace:
    error_type: str
    error_message: str
    timestamp: float
    
    primary_location: Optional[CodeLocation]
    call_stack: List[CodeLocation]
    
    execution_history: List[ExecutionEvent]
    related_files: Set[str]
    
    suspected_cause: str = ""
    confidence: float = 0.0
    trace_hash: str = ""
    
    def __post_init__(self):
        self.trace_hash = hashlib.md5(
            f"{self.error_type}:{self.error_message[:50]}:{self.primary_location}".encode()
        ).hexdigest()[:12]


class ErrorTracer:
    __slots__ = ("_events", "_active_ops", "_file_access", "_code_cache")
    
    def __init__(self):
        self._events: deque[ExecutionEvent] = deque(maxlen=200)
        self._active_ops: Dict[str, float] = {}
        self._file_access: Dict[str, List[float]] = {}
        self._code_cache: Dict[str, List[str]] = {}
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    
    def record_event(self, event_type: str, details: Dict[str, Any] = None, location: CodeLocation = None):
        self._events.append(ExecutionEvent(
            timestamp=time.time(),
            event_type=event_type,
            details=details or {},
            location=location,
        ))
    
    def record_file_access(self, file_path: str):
        if file_path not in self._file_access:
            self._file_access[file_path] = []
        self._file_access[file_path].append(time.time())
        self._file_access[file_path] = self._file_access[file_path][-10:]
    
    def start_operation(self, op_name: str):
        self._active_ops[op_name] = time.time()
        self.record_event("op_start", {"operation": op_name})
    
    def end_operation(self, op_name: str, success: bool = True, error: str = None):
        start = self._active_ops.pop(op_name, None)
        dt = (time.time() - start) * 1000 if start else 0
        self.record_event("op_end", {
            "operation": op_name,
            "success": success,
            "duration_ms": dt,
            "error": error[:100] if error else None,
        })
    
    def _get_code_lines(self, file_path: str) -> List[str]:
        if file_path in self._code_cache:
            return self._code_cache[file_path]
        try:
            lines = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
            self._code_cache[file_path] = lines
            return lines
        except Exception:
            return []
    
    def _extract_code_context(self, file_path: str, line: int, window: int = 5) -> str:
        lines = self._get_code_lines(file_path)
        if not lines:
            return ""
        start = max(0, line - window - 1)
        end = min(len(lines), line + window)
        return "\n".join(f"{i+1:4d}| {lines[i]}" for i in range(start, end))
    
    def _parse_traceback(self, tb_str: str) -> List[CodeLocation]:
        locations = []
        pattern = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
        for match in pattern.finditer(tb_str):
            file_path = match.group(1)
            line = int(match.group(2))
            func = match.group(3)
            snippet = self._extract_code_context(file_path, line, 3)
            locations.append(CodeLocation(file_path, line, func, snippet))
        return locations
    
    def _find_related_files(self, locations: List[CodeLocation]) -> Set[str]:
        related = set()
        for loc in locations:
            if "site-packages" not in loc.file and "<" not in loc.file:
                related.add(loc.file)
                try:
                    lines = self._get_code_lines(loc.file)
                    for line in lines[:50]:
                        if line.strip().startswith("from ") or line.strip().startswith("import "):
                            match = re.search(r"from (\S+) import|import (\S+)", line)
                            if match:
                                mod = match.group(1) or match.group(2)
                                if mod.startswith("jinx"):
                                    mod_path = mod.replace(".", "/") + ".py"
                                    related.add(mod_path)
                except Exception:
                    pass
        return related
    
    def _analyze_cause(self, error_type: str, error_msg: str, locations: List[CodeLocation]) -> Tuple[str, float]:
        causes = []
        confidence = 0.5
        
        if error_type == "ModuleNotFoundError":
            match = re.search(r"No module named ['\"]?(\w+)", error_msg)
            if match:
                module = match.group(1)
                causes.append(f"Missing module '{module}' - need optional import")
                confidence = 0.9
        
        elif error_type == "AttributeError":
            if "read-only" in error_msg:
                causes.append("Attempting to set attribute on __slots__ class")
                confidence = 0.85
            elif "has no attribute" in error_msg:
                match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_msg)
                if match:
                    causes.append(f"Missing attribute '{match.group(2)}' on {match.group(1)}")
                    confidence = 0.7
        
        elif error_type == "ImportError":
            if "circular" in error_msg.lower():
                causes.append("Circular import detected")
                confidence = 0.8
            else:
                causes.append("Import path or module structure issue")
                confidence = 0.6
        
        elif error_type == "SyntaxError":
            causes.append("Syntax error in source code")
            confidence = 0.95
        
        elif error_type == "TypeError":
            if "argument" in error_msg:
                causes.append("Function call with wrong arguments")
                confidence = 0.7
        
        if not causes and locations:
            primary = locations[-1]
            causes.append(f"Error in {primary.function}() at {Path(primary.file).name}:{primary.line}")
            confidence = 0.4
        
        return "; ".join(causes) if causes else "Unknown cause", confidence
    
    def trace_error(self, exception: BaseException, tb_str: str = "") -> ErrorTrace:
        if not tb_str:
            import traceback
            tb_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        
        error_type = type(exception).__name__
        error_msg = str(exception)
        
        locations = self._parse_traceback(tb_str)
        
        primary = None
        for loc in reversed(locations):
            if "site-packages" not in loc.file and "<" not in loc.file:
                primary = loc
                break
        
        related = self._find_related_files(locations)
        
        cause, confidence = self._analyze_cause(error_type, error_msg, locations)
        
        recent_events = list(self._events)[-30:]
        
        trace = ErrorTrace(
            error_type=error_type,
            error_message=error_msg,
            timestamp=time.time(),
            primary_location=primary,
            call_stack=locations,
            execution_history=recent_events,
            related_files=related,
            suspected_cause=cause,
            confidence=confidence,
        )
        
        self._save_trace(trace)
        self._notify_systems(trace)
        
        return trace
    
    def _save_trace(self, trace: ErrorTrace):
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = _TRACE_DIR / f"trace_{ts}_{trace.trace_hash}.json"
            
            data = {
                "error_type": trace.error_type,
                "error_message": trace.error_message,
                "timestamp": trace.timestamp,
                "trace_hash": trace.trace_hash,
                "suspected_cause": trace.suspected_cause,
                "confidence": trace.confidence,
                "primary_location": {
                    "file": trace.primary_location.file,
                    "line": trace.primary_location.line,
                    "function": trace.primary_location.function,
                } if trace.primary_location else None,
                "call_stack": [
                    {"file": l.file, "line": l.line, "function": l.function}
                    for l in trace.call_stack
                ],
                "related_files": list(trace.related_files),
                "recent_events": [
                    {"type": e.event_type, "details": e.details}
                    for e in trace.execution_history[-20:]
                ],
            }
            
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    
    def _notify_systems(self, trace: ErrorTrace):
        try:
            from jinx.micro.brain import learn_from_error
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(learn_from_error(
                    trace.primary_location.code_snippet if trace.primary_location else "",
                    trace.error_type,
                    trace.error_message,
                ))
            except RuntimeError:
                pass
        except Exception:
            pass
        
        try:
            from jinx.evolve import record_change
            if trace.primary_location:
                record_change(
                    trace.primary_location.file,
                    f"error:{trace.error_type}",
                    0,
                    False,
                )
        except Exception:
            pass
    
    def get_repair_context(self, trace: ErrorTrace) -> Dict[str, Any]:
        return {
            "error_type": trace.error_type,
            "error_message": trace.error_message,
            "suspected_cause": trace.suspected_cause,
            "confidence": trace.confidence,
            "primary_file": trace.primary_location.file if trace.primary_location else None,
            "primary_line": trace.primary_location.line if trace.primary_location else 0,
            "code_context": trace.primary_location.code_snippet if trace.primary_location else "",
            "related_files": list(trace.related_files),
            "recent_operations": [
                e.details.get("operation", e.event_type)
                for e in trace.execution_history
                if e.event_type in ("op_start", "op_end")
            ][-10:],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "events_recorded": len(self._events),
            "files_accessed": len(self._file_access),
            "active_operations": list(self._active_ops.keys()),
            "cached_files": len(self._code_cache),
        }


_tracer: Optional[ErrorTracer] = None


def get_error_tracer() -> ErrorTracer:
    global _tracer
    if _tracer is None:
        _tracer = ErrorTracer()
    return _tracer


def trace_error(exception: BaseException, tb_str: str = "") -> ErrorTrace:
    return get_error_tracer().trace_error(exception, tb_str)


def record_event(event_type: str, details: Dict[str, Any] = None):
    get_error_tracer().record_event(event_type, details)


def start_operation(op_name: str):
    get_error_tracer().start_operation(op_name)


def end_operation(op_name: str, success: bool = True, error: str = None):
    get_error_tracer().end_operation(op_name, success, error)


__all__ = [
    "ErrorTracer",
    "ErrorTrace",
    "CodeLocation",
    "get_error_tracer",
    "trace_error",
    "record_event",
    "start_operation",
    "end_operation",
]
