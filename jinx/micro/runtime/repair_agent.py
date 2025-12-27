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

_STATE_DIR: Path = Path(".jinx/repair")
_MAX_ITERATIONS: int = 5
_MAX_CONV_TURNS: int = 10


class AgentState(IntEnum):
    IDLE = 0
    ANALYZING = 1
    PLANNING = 2
    EXECUTING = 3
    VERIFYING = 4
    LEARNING = 5
    FAILED = 6
    SUCCESS = 7


@dataclass(slots=True)
class FileContext:
    path: Path
    content: str
    imports: List[str]
    classes: List[str]
    functions: List[str]
    dependencies: Set[str]
    hash: str


@dataclass(slots=True)
class ErrorContext:
    exception_type: str
    message: str
    traceback_raw: str
    frames: List[Dict[str, Any]]
    primary_file: Optional[Path]
    primary_line: int
    related_files: Set[Path]
    root_cause_hypothesis: str = ""


@dataclass(slots=True)
class RepairPlan:
    steps: List[Dict[str, Any]]
    affected_files: Set[Path]
    risk_level: str
    confidence: float
    rationale: str


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str
    timestamp: float


class ProjectAnalyzer:
    __slots__ = ("_root", "_file_cache", "_import_graph")

    def __init__(self, root: Path) -> None:
        self._root = root
        self._file_cache: Dict[Path, FileContext] = {}
        self._import_graph: Dict[str, Set[str]] = {}

    def analyze_file(self, path: Path) -> Optional[FileContext]:
        if path in self._file_cache:
            cached = self._file_cache[path]
            current_hash = hashlib.md5(path.read_bytes()).hexdigest()
            if cached.hash == current_hash:
                return cached

        if not path.exists() or not path.suffix == ".py":
            return None

        content = path.read_text(encoding="utf-8", errors="replace")
        file_hash = hashlib.md5(content.encode()).hexdigest()

        imports: List[str] = []
        classes: List[str] = []
        functions: List[str] = []
        deps: Set[str] = set()

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                        deps.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        deps.add(node.module.split(".")[0])
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    functions.append(node.name)
        except SyntaxError:
            pass

        ctx = FileContext(
            path=path,
            content=content,
            imports=imports,
            classes=classes,
            functions=functions,
            dependencies=deps,
            hash=file_hash,
        )
        self._file_cache[path] = ctx
        return ctx

    def find_related_files(self, target: Path, max_files: int = 5) -> Set[Path]:
        related: Set[Path] = set()
        target_ctx = self.analyze_file(target)
        if not target_ctx:
            return related

        try:
            target_module = str(target.relative_to(self._root)).replace("/", ".").replace("\\", ".").rstrip(".py")
        except ValueError:
            return related

        checked = 0
        for py_file in self._root.rglob("*.py"):
            if len(related) >= max_files or checked > 50:
                break
            if py_file == target or "__pycache__" in str(py_file):
                continue
            checked += 1
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")[:5000]
                if target_module in content or any(c in content for c in target_ctx.classes[:3]):
                    related.add(py_file)
            except Exception:
                continue

        return related

    def get_project_structure(self) -> str:
        lines = []
        count = 0
        for py_file in sorted(self._root.rglob("*.py")):
            if count >= 30 or "__pycache__" in str(py_file):
                continue
            try:
                rel = py_file.relative_to(self._root)
                lines.append(str(rel))
                count += 1
            except Exception:
                continue
        return "\n".join(lines[:30])


class LLMInterface:
    __slots__ = ("_api_key", "_model", "_conversation", "_enabled")

    def __init__(self) -> None:
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._conversation: Deque[ConversationTurn] = deque(maxlen=20)
        self._enabled = bool(self._api_key)

    def _call_api(self, messages: List[Dict[str, str]], max_tokens: int = 4000) -> Optional[str]:
        if not self._enabled:
            return None

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            data = json.dumps({
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }).encode()

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=data,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode())

            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [LLM Error: {e}]")
            return None

    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> Optional[str]:
        self._conversation.append(ConversationTurn("user", user_message, time.time()))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in self._conversation:
            messages.append({"role": turn.role, "content": turn.content})

        response = self._call_api(messages)
        if response:
            self._conversation.append(ConversationTurn("assistant", response, time.time()))

        return response

    def reset_conversation(self) -> None:
        self._conversation.clear()


class RepairExecutor:
    __slots__ = ("_backups",)

    def __init__(self) -> None:
        self._backups: Dict[Path, str] = {}

    def backup(self, path: Path) -> None:
        if path.exists():
            self._backups[path] = path.read_text(encoding="utf-8")

    def restore_all(self) -> None:
        for path, content in self._backups.items():
            path.write_text(content, encoding="utf-8")
        self._backups.clear()

    def apply_edit(self, path: Path, old_code: str, new_code: str) -> bool:
        if not path.exists():
            return False

        self.backup(path)
        content = path.read_text(encoding="utf-8")

        if old_code in content:
            new_content = content.replace(old_code, new_code, 1)
            path.write_text(new_content, encoding="utf-8")
            return True

        lines = content.split("\n")
        old_lines = old_code.strip().split("\n")
        for i in range(len(lines) - len(old_lines) + 1):
            chunk = "\n".join(lines[i:i + len(old_lines)])
            if chunk.strip() == old_code.strip():
                lines[i:i + len(old_lines)] = new_code.split("\n")
                path.write_text("\n".join(lines), encoding="utf-8")
                return True

        return False

    def create_file(self, path: Path, content: str) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True

    def verify_syntax(self, path: Path) -> Tuple[bool, str]:
        try:
            content = path.read_text(encoding="utf-8")
            ast.parse(content)
            return True, ""
        except SyntaxError as e:
            return False, str(e)


class CognitiveRepairAgent:
    __slots__ = (
        "_state", "_analyzer", "_llm", "_executor", "_error_ctx",
        "_plan", "_iteration", "_history", "_project_root"
    )

    SYSTEM_PROMPT = """You are Jinx's self-repair cognitive agent. You analyze Python errors and generate precise fixes.

CAPABILITIES:
- Deep code analysis across multiple files
- Understanding of import dependencies and class hierarchies  
- Iterative repair with verification
- Learning from past repairs

RESPONSE FORMAT for analysis:
<root_cause>Precise explanation of the bug</root_cause>
<affected_files>file1.py, file2.py</affected_files>
<repair_plan>
Step 1: [action] in [file] - [description]
Step 2: ...
</repair_plan>
<confidence>0.0-1.0</confidence>

RESPONSE FORMAT for code fix:
<file>path/to/file.py</file>
<old_code>
exact code to replace
</old_code>
<new_code>
fixed code
</new_code>
<explanation>Why this fixes the issue</explanation>

RULES:
1. Always preserve original code style and indentation
2. Never remove functionality, only fix bugs
3. Consider side effects on other files
4. If unsure, ask for more context
5. Prefer minimal changes over rewrites"""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self._state = AgentState.IDLE
        self._project_root = project_root or Path.cwd()
        self._analyzer = ProjectAnalyzer(self._project_root)
        self._llm = LLMInterface()
        self._executor = RepairExecutor()
        self._error_ctx: Optional[ErrorContext] = None
        self._plan: Optional[RepairPlan] = None
        self._iteration = 0
        self._history: List[Dict] = []

    def _parse_traceback(self, tb_str: str) -> List[Dict[str, Any]]:
        frames = []
        pattern = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
        for match in pattern.finditer(tb_str):
            frames.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "function": match.group(3),
            })
        return frames

    def _extract_code_context(self, path: Path, line: int, window: int = 20) -> str:
        try:
            lines = path.read_text(encoding="utf-8").split("\n")
            start = max(0, line - window)
            end = min(len(lines), line + window)
            numbered = [f"{i+1:4d} | {lines[i]}" for i in range(start, end)]
            return "\n".join(numbered)
        except Exception:
            return ""

    def _build_analysis_prompt(self, error_ctx: ErrorContext) -> str:
        project_struct = self._analyzer.get_project_structure()

        primary_context = ""
        if error_ctx.primary_file:
            primary_context = self._extract_code_context(
                error_ctx.primary_file, 
                error_ctx.primary_line,
                window=30
            )

        related_contexts = []
        for rf in list(error_ctx.related_files)[:3]:
            ctx = self._analyzer.analyze_file(rf)
            if ctx:
                related_contexts.append(f"=== {rf} ===\n{ctx.content[:2000]}")

        return f"""ANALYZE THIS ERROR AND CREATE A REPAIR PLAN:

EXCEPTION: {error_ctx.exception_type}
MESSAGE: {error_ctx.message}

TRACEBACK:
{error_ctx.traceback_raw}

PRIMARY FILE ({error_ctx.primary_file}:{error_ctx.primary_line}):
```python
{primary_context}
```

RELATED FILES:
{chr(10).join(related_contexts)}

PROJECT STRUCTURE:
{project_struct[:3000]}

Analyze the root cause and provide a repair plan."""

    def _build_fix_prompt(self, step: Dict[str, Any]) -> str:
        file_path = Path(step.get("file", ""))
        ctx = self._analyzer.analyze_file(file_path)
        content = ctx.content if ctx else ""
        action = step.get("action", "")
        module = step.get("module", "")

        if action == "make_import_optional" and module:
            return f"""Make the import of '{module}' optional in this file.

FILE: {file_path}

CURRENT CONTENT:
```python
{content}
```

Find the line "import {module}" or "from {module} import ..." and wrap it in try/except.

Respond EXACTLY in this format:
<file>{file_path}</file>
<old_code>
import {module}
</old_code>
<new_code>
try:
    import {module}
    _HAS_{module.upper()} = True
except ImportError:
    _HAS_{module.upper()} = False
    {module} = None  # type: ignore
</new_code>
<explanation>Made {module} import optional</explanation>"""

        return f"""GENERATE THE EXACT CODE FIX:

FILE: {file_path}
ACTION: {action}
DESCRIPTION: {step.get('description', '')}

CURRENT FILE CONTENT:
```python
{content}
```

ERROR CONTEXT: {self._error_ctx.message if self._error_ctx else ''}

Respond in this format:
<file>path/to/file.py</file>
<old_code>exact original code</old_code>
<new_code>fixed code</new_code>
<explanation>why this fixes it</explanation>"""

    def _parse_analysis_response(self, response: str) -> Optional[RepairPlan]:
        root_cause = re.search(r"<root_cause>(.*?)</root_cause>", response, re.DOTALL)
        affected = re.search(r"<affected_files>(.*?)</affected_files>", response, re.DOTALL)
        plan_match = re.search(r"<repair_plan>(.*?)</repair_plan>", response, re.DOTALL)
        conf_match = re.search(r"<confidence>([\d.]+)</confidence>", response)

        steps = []
        
        if plan_match:
            plan_text = plan_match.group(1).strip()
            for line in plan_text.split("\n"):
                step_match = re.match(r"Step \d+:\s*\[(\w+)\]\s*in\s*\[([^\]]+)\]\s*-\s*(.*)", line.strip())
                if step_match:
                    steps.append({
                        "action": step_match.group(1),
                        "file": step_match.group(2),
                        "description": step_match.group(3),
                    })
        
        if not steps and self._error_ctx:
            steps = self._generate_direct_fix_steps()

        affected_files = set()
        if affected:
            for f in affected.group(1).split(","):
                f = f.strip()
                if f:
                    affected_files.add(self._project_root / f)

        return RepairPlan(
            steps=steps,
            affected_files=affected_files,
            risk_level="medium",
            confidence=float(conf_match.group(1)) if conf_match else 0.5,
            rationale=root_cause.group(1).strip() if root_cause else response[:200],
        )

    def _generate_direct_fix_steps(self) -> List[Dict[str, Any]]:
        if not self._error_ctx:
            return []
        
        steps = []
        msg = self._error_ctx.message
        exc_type = self._error_ctx.exception_type
        
        if exc_type == "ModuleNotFoundError" or "No module named" in msg:
            module_match = re.search(r"No module named ['\"]?(\w+)['\"]?", msg)
            if module_match and self._error_ctx.primary_file:
                module_name = module_match.group(1)
                steps.append({
                    "action": "make_import_optional",
                    "file": str(self._error_ctx.primary_file),
                    "description": f"Make import of '{module_name}' optional with try/except",
                    "module": module_name,
                })
        
        elif exc_type == "AttributeError" and self._error_ctx.primary_file:
            steps.append({
                "action": "fix_attribute",
                "file": str(self._error_ctx.primary_file),
                "description": f"Fix attribute error: {msg[:50]}",
            })
        
        elif exc_type == "ImportError" and self._error_ctx.primary_file:
            steps.append({
                "action": "fix_import",
                "file": str(self._error_ctx.primary_file),
                "description": f"Fix import error: {msg[:50]}",
            })
        
        elif self._error_ctx.primary_file:
            steps.append({
                "action": "fix_error",
                "file": str(self._error_ctx.primary_file),
                "description": f"Fix {exc_type}: {msg[:50]}",
            })
        
        return steps

    def _parse_fix_response(self, response: str) -> Optional[Tuple[Path, str, str, str]]:
        file_match = re.search(r"<file>(.*?)</file>", response, re.DOTALL)
        old_match = re.search(r"<old_code>(.*?)</old_code>", response, re.DOTALL)
        new_match = re.search(r"<new_code>(.*?)</new_code>", response, re.DOTALL)
        exp_match = re.search(r"<explanation>(.*?)</explanation>", response, re.DOTALL)

        if not all([file_match, old_match, new_match]):
            return None

        return (
            Path(file_match.group(1).strip()),
            old_match.group(1).strip(),
            new_match.group(1).strip(),
            exp_match.group(1).strip() if exp_match else "",
        )

    def _verify_repair(self) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import jinx"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self._project_root),
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr
        except Exception as e:
            return False, str(e)

    def analyze_error(self, exception: BaseException, tb_str: str = "") -> ErrorContext:
        self._state = AgentState.ANALYZING
        print("  📊 Analyzing error context...")

        if not tb_str:
            tb_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        trace_ctx = None
        try:
            from jinx.micro.runtime.error_tracer import trace_error
            trace = trace_error(exception, tb_str)
            trace_ctx = {
                "suspected_cause": trace.suspected_cause,
                "confidence": trace.confidence,
                "recent_ops": [e.event_type for e in trace.execution_history[-10:]],
            }
            print(f"  🔍 Traced: {trace.suspected_cause} (conf: {trace.confidence:.0%})")
        except Exception:
            pass

        frames = self._parse_traceback(tb_str)

        primary_file = None
        primary_line = 0
        for frame in reversed(frames):
            fp = Path(frame["file"])
            if fp.exists() and "site-packages" not in str(fp) and "<" not in str(fp):
                primary_file = fp
                primary_line = frame["line"]
                break

        related = set()
        if primary_file:
            related = self._analyzer.find_related_files(primary_file)

        root_cause = ""
        if trace_ctx:
            root_cause = trace_ctx.get("suspected_cause", "")

        self._error_ctx = ErrorContext(
            exception_type=type(exception).__name__,
            message=str(exception),
            traceback_raw=tb_str,
            frames=frames,
            primary_file=primary_file,
            primary_line=primary_line,
            related_files=related,
            root_cause_hypothesis=root_cause,
        )

        return self._error_ctx

    def create_plan(self) -> Optional[RepairPlan]:
        if not self._error_ctx:
            return None

        self._state = AgentState.PLANNING
        print("  🧠 Creating repair plan with AI...")

        prompt = self._build_analysis_prompt(self._error_ctx)
        response = self._llm.chat(prompt, self.SYSTEM_PROMPT)

        if not response:
            print("  ❌ LLM not available")
            return None

        self._plan = self._parse_analysis_response(response)
        if self._plan:
            print(f"  ✓ Plan created: {len(self._plan.steps)} steps, confidence {self._plan.confidence:.0%}")
            print(f"  📝 Root cause: {self._plan.rationale[:100]}...")

        return self._plan

    def _try_direct_fix(self, step: Dict[str, Any]) -> Optional[Tuple[Path, str, str, str]]:
        action = step.get("action", "")
        file_path = Path(step.get("file", ""))
        module = step.get("module", "")
        
        if not file_path.exists():
            return None
        
        content = file_path.read_text(encoding="utf-8")
        
        if action == "make_import_optional" and module:
            for pattern in [f"import {module}", f"from {module} import"]:
                if pattern in content:
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if pattern in line and not line.strip().startswith("#"):
                            old_code = line
                            indent = len(line) - len(line.lstrip())
                            sp = " " * indent
                            new_code = f"""{sp}try:
{sp}    {line.strip()}
{sp}    _HAS_{module.upper()} = True
{sp}except ImportError:
{sp}    _HAS_{module.upper()} = False
{sp}    {module} = None  # type: ignore"""
                            return (file_path, old_code, new_code, f"Made {module} import optional")
        return None

    def execute_plan(self) -> bool:
        if not self._plan or not self._plan.steps:
            return False

        self._state = AgentState.EXECUTING
        print(f"  🔧 Executing {len(self._plan.steps)} repair steps...")

        for i, step in enumerate(self._plan.steps):
            print(f"    Step {i+1}: {step.get('action')} in {step.get('file')}")

            direct_fix = self._try_direct_fix(step)
            if direct_fix:
                file_path, old_code, new_code, explanation = direct_fix
                print(f"    📄 Direct fix: {explanation[:50]}...")
            else:
                fix_prompt = self._build_fix_prompt(step)
                response = self._llm.chat(fix_prompt)

                if not response:
                    print(f"    ❌ Failed to get fix for step {i+1}")
                    continue

                fix = self._parse_fix_response(response)
                if not fix:
                    print(f"    ❌ Could not parse fix response")
                    continue

                file_path, old_code, new_code, explanation = fix
                print(f"    📄 Applying fix: {explanation[:50]}...")

            if self._executor.apply_edit(file_path, old_code, new_code):
                valid, error = self._executor.verify_syntax(file_path)
                if valid:
                    print(f"    ✓ Step {i+1} applied successfully")
                else:
                    print(f"    ❌ Syntax error after fix: {error}")
                    self._executor.restore_all()
                    return False
            else:
                print(f"    ⚠️ Could not apply edit, asking LLM for alternatives...")
                
                retry_prompt = f"""The previous fix could not be applied because the old_code was not found exactly.

Current file content:
```python
{Path(step.get('file', '')).read_text(encoding='utf-8') if Path(step.get('file', '')).exists() else 'FILE NOT FOUND'}
```

Please provide a more precise old_code that matches exactly."""
                
                retry_response = self._llm.chat(retry_prompt)
                if retry_response:
                    retry_fix = self._parse_fix_response(retry_response)
                    if retry_fix:
                        _, old2, new2, _ = retry_fix
                        if self._executor.apply_edit(file_path, old2, new2):
                            print(f"    ✓ Step {i+1} applied on retry")
                        else:
                            print(f"    ❌ Retry also failed")

        return True

    def verify(self) -> Tuple[bool, str]:
        self._state = AgentState.VERIFYING
        print("  🔍 Verifying repair...")

        success, error = self._verify_repair()
        if success:
            print("  ✓ Verification passed!")
            self._state = AgentState.SUCCESS
        else:
            print(f"  ❌ Verification failed: {error[:200]}")
            self._state = AgentState.FAILED

        return success, error

    def repair(self, exception: BaseException, tb_str: str = "") -> bool:
        print(f"\n{'='*60}")
        print("🤖 JINX COGNITIVE REPAIR AGENT")
        print(f"{'='*60}")
        print(f"Error: {exception}")

        if not self._llm._enabled:
            print("\n⚠️ OpenAI API key not set - AI repair disabled")
            print("   Set OPENAI_API_KEY for intelligent repairs")
            return False

        self._llm.reset_conversation()
        self._iteration = 0

        while self._iteration < _MAX_ITERATIONS:
            self._iteration += 1
            print(f"\n--- Iteration {self._iteration}/{_MAX_ITERATIONS} ---")

            self.analyze_error(exception, tb_str)
            plan = self.create_plan()

            if not plan:
                print("  Could not create repair plan")
                continue

            if not self.execute_plan():
                print("  Plan execution failed")
                self._executor.restore_all()
                continue

            success, error = self.verify()
            if success:
                self._learn_success()
                return True
            else:
                print(f"  Refining approach based on new error...")
                self._executor.restore_all()
                
                refine_prompt = f"""The repair was applied but verification failed with a new error:

{error}

Please analyze this new error and adjust the repair plan."""
                self._llm.chat(refine_prompt)

        print(f"\n❌ Could not repair after {_MAX_ITERATIONS} iterations")
        self._state = AgentState.FAILED
        return False

    def _learn_success(self) -> None:
        self._state = AgentState.LEARNING
        if self._error_ctx and self._plan:
            self._history.append({
                "error_type": self._error_ctx.exception_type,
                "message": self._error_ctx.message[:200],
                "solution": self._plan.rationale,
                "steps": len(self._plan.steps),
                "iterations": self._iteration,
                "timestamp": time.time(),
            })
            self._save_history()

    def _save_history(self) -> None:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        history_file = _STATE_DIR / "repair_history.json"
        try:
            history_file.write_text(json.dumps(self._history[-100:], indent=2))
        except Exception:
            pass


_agent: Optional[CognitiveRepairAgent] = None


def get_repair_agent() -> CognitiveRepairAgent:
    global _agent
    if _agent is None:
        _agent = CognitiveRepairAgent()
    return _agent


def cognitive_repair(exception: BaseException, tb_str: str = "") -> bool:
    agent = get_repair_agent()
    return agent.repair(exception, tb_str)


def auto_repair_and_restart(exception: BaseException, restart_cmd: Optional[List[str]] = None) -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️ OPENAI_API_KEY not set - cannot perform AI repair")
        return False

    success = cognitive_repair(exception)

    if success and restart_cmd:
        print(f"\n🔄 Restarting Jinx...")
        time.sleep(0.5)
        os.execv(sys.executable, [sys.executable] + restart_cmd)

    return success


__all__ = [
    "CognitiveRepairAgent",
    "get_repair_agent", 
    "cognitive_repair",
    "auto_repair_and_restart",
]
