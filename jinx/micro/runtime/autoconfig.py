from __future__ import annotations

import json
import math
import os
import time
from array import array
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

_PSUTIL: Any = None
try:
    import psutil as _psutil_mod
    _PSUTIL = _psutil_mod
except ImportError:
    pass

_FALSE: Final[frozenset] = frozenset({"", "0", "false", "off", "no"})
_CONFIG_PATH: Final[Path] = Path(".jinx/autoconfig_state.json")


class PerformanceTier(IntEnum):
    MINIMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4


class ConfigDomain(IntEnum):
    CORE = auto()
    BRAIN = auto()
    MEMORY = auto()
    EMBEDDINGS = auto()
    RUNTIME = auto()
    LLM = auto()
    UI = auto()


@dataclass(slots=True)
class SystemProfile:
    cpu_count: int = 4
    cpu_freq_mhz: float = 2000.0
    memory_gb: float = 8.0
    disk_speed_mb_s: float = 100.0
    gpu_available: bool = False
    tier: PerformanceTier = PerformanceTier.MEDIUM
    
    @classmethod
    def detect(cls) -> "SystemProfile":
        profile = cls()
        if _PSUTIL:
            profile.cpu_count = _PSUTIL.cpu_count(logical=True) or 4
            profile.memory_gb = _PSUTIL.virtual_memory().total / (1024 ** 3)
            freq = _PSUTIL.cpu_freq()
            if freq:
                profile.cpu_freq_mhz = freq.current or 2000.0
        
        score = (
            min(1.0, profile.cpu_count / 16) * 0.3 +
            min(1.0, profile.memory_gb / 32) * 0.4 +
            min(1.0, profile.cpu_freq_mhz / 4000) * 0.3
        )
        
        if score >= 0.8:
            profile.tier = PerformanceTier.EXTREME
        elif score >= 0.6:
            profile.tier = PerformanceTier.HIGH
        elif score >= 0.4:
            profile.tier = PerformanceTier.MEDIUM
        elif score >= 0.2:
            profile.tier = PerformanceTier.LOW
        else:
            profile.tier = PerformanceTier.MINIMAL
        
        return profile


@dataclass(slots=True)
class ConfigParam:
    name: str
    value: str
    domain: ConfigDomain
    tier_values: Dict[PerformanceTier, str] = field(default_factory=dict)
    adaptive: bool = False
    last_update: float = 0.0


class ConfigHistory:
    __slots__ = ("_changes", "_max_size")
    
    def __init__(self, max_size: int = 1000) -> None:
        self._changes: List[Tuple[float, str, str, str]] = []
        self._max_size = max_size
    
    def record(self, name: str, old: str, new: str) -> None:
        self._changes.append((time.monotonic(), name, old, new))
        if len(self._changes) > self._max_size:
            self._changes = self._changes[-self._max_size // 2:]
    
    def recent(self, n: int = 10) -> List[Tuple[float, str, str, str]]:
        return self._changes[-n:]


class AdaptiveOptimizer:
    __slots__ = ("_samples", "_rewards", "_alpha", "_params")
    
    def __init__(self) -> None:
        self._samples: Dict[str, array] = {}
        self._rewards: Dict[str, array] = {}
        self._alpha = 0.1
        self._params: Dict[str, float] = {}
    
    def observe(self, param: str, value: float, reward: float) -> None:
        if param not in self._samples:
            self._samples[param] = array("d", [])
            self._rewards[param] = array("d", [])
        self._samples[param].append(value)
        self._rewards[param].append(reward)
        
        if len(self._samples[param]) > 100:
            self._samples[param] = self._samples[param][-50:]
            self._rewards[param] = self._rewards[param][-50:]
    
    def suggest(self, param: str, current: float, bounds: Tuple[float, float]) -> float:
        if param not in self._samples or len(self._samples[param]) < 5:
            return current
        
        samples = list(self._samples[param])
        rewards = list(self._rewards[param])
        
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
        best_val = samples[best_idx]
        
        direction = best_val - current
        step = direction * self._alpha
        
        new_val = current + step
        return max(bounds[0], min(bounds[1], new_val))


class CognitiveConfigurator:
    __slots__ = (
        "_profile", "_params", "_history", "_optimizer", "_state_path",
        "_boot_time", "_applied", "_tier_overrides"
    )
    
    def __init__(self) -> None:
        self._profile = SystemProfile.detect()
        self._params: Dict[str, ConfigParam] = {}
        self._history = ConfigHistory()
        self._optimizer = AdaptiveOptimizer()
        self._state_path = _CONFIG_PATH
        self._boot_time = time.monotonic()
        self._applied = False
        self._tier_overrides: Dict[str, PerformanceTier] = {}
    
    def _load_state(self) -> None:
        if self._state_path.exists():
            data = json.loads(self._state_path.read_text())
            for name, info in data.get("params", {}).items():
                if name in self._params:
                    self._params[name].value = info.get("value", self._params[name].value)
                    self._params[name].last_update = info.get("last_update", 0.0)
    
    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "params": {
                name: {"value": p.value, "last_update": p.last_update}
                for name, p in self._params.items()
            },
            "profile": {
                "tier": int(self._profile.tier),
                "cpu_count": self._profile.cpu_count,
                "memory_gb": self._profile.memory_gb,
            },
            "boot_time": self._boot_time,
        }
        self._state_path.write_text(json.dumps(data, indent=2))
    
    def _set(self, name: str, value: str, domain: ConfigDomain = ConfigDomain.CORE) -> None:
        old = os.environ.get(name, "")
        if old and old != value:
            self._history.record(name, old, value)
        os.environ[name] = value
        
        if name not in self._params:
            self._params[name] = ConfigParam(name=name, value=value, domain=domain)
        else:
            self._params[name].value = value
        self._params[name].last_update = time.monotonic()
    
    def _set_if_empty(self, name: str, value: str, domain: ConfigDomain = ConfigDomain.CORE) -> None:
        if not os.environ.get(name):
            self._set(name, value, domain)
    
    def _tier_value(self, base: int, multipliers: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)) -> str:
        idx = min(int(self._profile.tier), len(multipliers) - 1)
        return str(int(base * multipliers[idx]))
    
    def _tier_bool(self, threshold: PerformanceTier = PerformanceTier.MEDIUM) -> str:
        return "1" if self._profile.tier >= threshold else "0"


_configurator: Optional[CognitiveConfigurator] = None


def _get_configurator() -> CognitiveConfigurator:
    global _configurator
    if _configurator is None:
        _configurator = CognitiveConfigurator()
    return _configurator


def apply_auto_defaults(settings: Any | None = None) -> None:
    if os.environ.get("JINX_AUTO_MODE", "1").lower() in _FALSE:
        return
    
    cfg = _get_configurator()
    cfg._load_state()
    tier = cfg._profile.tier
    s = cfg._set_if_empty
    tv = cfg._tier_value
    tb = cfg._tier_bool
    
    # ═══════════════════════════════════════════════════════════════
    # CORE — autonomous operation baseline
    # ═══════════════════════════════════════════════════════════════
    s("OPENAI_MODEL", "gpt-4o", ConfigDomain.LLM)
    s("PULSE", tv(500, (250, 350, 500, 750, 1000)), ConfigDomain.CORE)
    s("TIMEOUT", tv(1000, (500, 750, 1000, 1500, 2000)), ConfigDomain.CORE)
    s("JINX_PULSE_HARD_SHUTDOWN", "0", ConfigDomain.CORE)
    s("JINX_CRASH_DIAGNOSTICS", "1", ConfigDomain.CORE)
    s("JINX_SELF_HEALING", "1", ConfigDomain.CORE)
    s("JINX_ML_SYSTEM", "1", ConfigDomain.CORE)
    s("JINX_DYNAMIC_CONFIG", "1", ConfigDomain.CORE)
    s("JINX_AUTO_ACTION", "1", ConfigDomain.CORE)
    s("JINX_FILE_PREVIEW", "1", ConfigDomain.CORE)
    s("JINX_LOCATOR_CONC", tv(3, (1, 2, 3, 4, 6)), ConfigDomain.RUNTIME)
    s("JINX_STAGE_BASECTX_MS", tv(500, (300, 400, 500, 600, 800)), ConfigDomain.RUNTIME)
    s("JINX_STAGE_PROJCTX_MS", tv(5000, (2000, 3500, 5000, 7000, 10000)), ConfigDomain.RUNTIME)
    s("JINX_STAGE_MEMCTX_MS", tv(500, (300, 400, 500, 600, 800)), ConfigDomain.RUNTIME)

    # ═══════════════════════════════════════════════════════════════
    # EMBEDDINGS — tier-scaled retrieval power
    # ═══════════════════════════════════════════════════════════════
    s("EMBED_PROJECT_ENABLE", "1", ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_TOP_K", tv(50, (20, 35, 50, 75, 100)), ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_EXHAUSTIVE", tb(PerformanceTier.MEDIUM), ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_NO_STAGE_BUDGETS", "0", ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_TOTAL_CODE_BUDGET", tv(50000, (20000, 35000, 50000, 75000, 100000)), ConfigDomain.EMBEDDINGS)
    s("EMBED_UNIFIED_MAX_TIME_MS", tv(3000, (1500, 2000, 3000, 4500, 6000)), ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_SCORE_THRESHOLD", "0.15", ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_CALLGRAPH", "1", ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_CALLGRAPH_TOP_HITS", tv(5, (2, 3, 5, 7, 10)), ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_CALLGRAPH_CALLERS_LIMIT", tv(5, (2, 3, 5, 7, 10)), ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_CALLGRAPH_CALLEES_LIMIT", tv(5, (2, 3, 5, 7, 10)), ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_ALWAYS_FULL_PY_SCOPE", "1", ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_FULL_SCOPE_TOP_N", tv(10, (5, 7, 10, 15, 20)), ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_SNIPPET_PER_HIT_CHARS", tv(2000, (1000, 1500, 2000, 3000, 4000)), ConfigDomain.EMBEDDINGS)
    s("EMBED_PROJECT_ROOT", ".", ConfigDomain.EMBEDDINGS)
    s("JINX_EMBED_MEMORY_CTX", "1", ConfigDomain.EMBEDDINGS)
    s("EMBED_BRAIN_ENABLE", "1", ConfigDomain.EMBEDDINGS)
    s("EMBED_BRAIN_TOP_K", tv(10, (5, 7, 10, 15, 20)), ConfigDomain.EMBEDDINGS)
    s("EMBED_BRAIN_EXPAND_MAX_TOKENS", tv(6, (3, 4, 6, 8, 10)), ConfigDomain.EMBEDDINGS)

    # ═══════════════════════════════════════════════════════════════
    # BRAIN SYSTEMS — all 27 cognitive modules
    # ═══════════════════════════════════════════════════════════════
    brain_modules = [
        "JINX_BRAIN_ENABLE", "JINX_BRAIN_AUTO_INIT", "JINX_BRAIN_ADAPTIVE_RETRIEVAL",
        "JINX_BRAIN_UCB1", "JINX_BRAIN_THRESHOLD_LEARNING", "JINX_BRAIN_THOMPSON_SAMPLING",
        "JINX_BRAIN_QUERY_CLASSIFICATION", "JINX_BRAIN_INTENT_LEARNING",
        "JINX_BRAIN_CONTEXT_OPTIMIZATION", "JINX_BRAIN_Q_LEARNING",
        "JINX_BRAIN_SEMANTIC_ROUTING", "JINX_BRAIN_INTELLIGENT_PLANNING",
        "JINX_BRAIN_CACHE_OPTIMIZATION", "JINX_BRAIN_OUTCOME_TRACKING",
        "JINX_BRAIN_PATTERN_RECOGNITION", "JINX_BRAIN_META_COGNITIVE",
        "JINX_BRAIN_GOAL_DRIVEN", "JINX_BRAIN_ENSEMBLE", "JINX_BRAIN_KNOWLEDGE_GRAPH",
        "JINX_BRAIN_QUERY_EXPANDER", "JINX_BRAIN_AUTO_TUNER", "JINX_BRAIN_PERFORMANCE_MONITOR",
    ]
    for mod in brain_modules:
        s(mod, "1", ConfigDomain.BRAIN)

    # ═══════════════════════════════════════════════════════════════
    # MEMORY SYSTEMS — 4 cognitive memory layers
    # ═══════════════════════════════════════════════════════════════
    s("JINX_MEMORY_WORKING", "1", ConfigDomain.MEMORY)
    s("JINX_MEMORY_WORKING_SIZE", tv(7, (5, 6, 7, 9, 12)), ConfigDomain.MEMORY)
    s("JINX_MEMORY_EPISODIC", "1", ConfigDomain.MEMORY)
    s("JINX_MEMORY_EPISODIC_SEARCH", "1", ConfigDomain.MEMORY)
    s("JINX_MEMORY_SEMANTIC", "1", ConfigDomain.MEMORY)
    s("JINX_PERSIST_MEMORY", "1", ConfigDomain.MEMORY)
    s("JINX_MEMORY_DIR", ".jinx/memory", ConfigDomain.MEMORY)
    s("JINX_MEMORY_INTEGRATION", "1", ConfigDomain.MEMORY)
    s("JINX_MEMORY_CONSOLIDATION", "1", ConfigDomain.MEMORY)

    # ═══════════════════════════════════════════════════════════════
    # AI EDITOR — code intelligence
    # ═══════════════════════════════════════════════════════════════
    editor_modules = [
        "JINX_AI_EDITOR_ENABLE", "JINX_AI_ANALYZER_ENABLE", "JINX_AI_DIAGNOSTICS",
        "JINX_AI_COMPLETIONS", "JINX_AI_SUGGESTIONS", "JINX_SEMANTIC_PATCH_ENABLE",
        "JINX_SEMANTIC_PATCH_ML", "JINX_AST_TRANSFORMS", "JINX_LIBCST_ENABLE",
        "JINX_CODE_SMELL_DETECTION", "JINX_PATTERN_LEARNING",
    ]
    for mod in editor_modules:
        s(mod, "1", ConfigDomain.LLM)

    # ═══════════════════════════════════════════════════════════════
    # RUNTIME — real-time parameters
    # ═══════════════════════════════════════════════════════════════
    s("JINX_CONTEXT_CONTINUITY", "1", ConfigDomain.RUNTIME)
    s("JINX_CONTEXT_HISTORY", tv(20, (10, 15, 20, 30, 50)), ConfigDomain.RUNTIME)
    s("JINX_SMART_CACHE_ENABLE", "1", ConfigDomain.RUNTIME)
    s("JINX_SMART_CACHE_MB", tv(100, (50, 75, 100, 150, 250)), ConfigDomain.RUNTIME)
    s("JINX_SMART_CACHE_ENTRIES", tv(1000, (500, 750, 1000, 1500, 2500)), ConfigDomain.RUNTIME)
    s("JINX_CHAINED_REASONING", "1", ConfigDomain.RUNTIME)
    s("JINX_CHAINED_REFLECT", "1", ConfigDomain.RUNTIME)
    s("JINX_CHAINED_ADVISORY", "1", ConfigDomain.RUNTIME)
    s("JINX_RUNTIME_USE_PRIORITY_QUEUE", "1", ConfigDomain.RUNTIME)
    s("JINX_LLM_STREAM_FASTPATH", "1", ConfigDomain.RUNTIME)
    s("JINX_COOP_YIELD", "1", ConfigDomain.RUNTIME)
    s("JINX_CTX_COMPACT", "1", ConfigDomain.RUNTIME)
    s("JINX_CTX_COMPACT_ORCH", "1", ConfigDomain.RUNTIME)
    s("EMBED_SLICE_MS", tv(12, (8, 10, 12, 15, 20)), ConfigDomain.RUNTIME)
    s("JINX_LOCATOR_VEC_MS", tv(120, (80, 100, 120, 150, 200)), ConfigDomain.RUNTIME)
    s("JINX_CPU_WORKERS", "0", ConfigDomain.RUNTIME)
    s("JINX_ADM_GRAPH_CONC", tv(1, (1, 1, 1, 2, 2)), ConfigDomain.RUNTIME)
    s("JINX_ADM_PATCH_CONC", tv(2, (1, 1, 2, 3, 4)), ConfigDomain.RUNTIME)
    s("JINX_ADM_LLM_CONC", tv(2, (1, 2, 2, 3, 4)), ConfigDomain.RUNTIME)
    s("JINX_ADM_TURN_CONC", tv(4, (2, 3, 4, 6, 8)), ConfigDomain.RUNTIME)

    # ═══════════════════════════════════════════════════════════════
    # LLM & CONSENSUS
    # ═══════════════════════════════════════════════════════════════
    s("JINX_LLM_CONSENSUS", tb(PerformanceTier.MEDIUM), ConfigDomain.LLM)
    s("JINX_LLM_CONSENSUS_MS", tv(500, (300, 400, 500, 700, 1000)), ConfigDomain.LLM)
    s("JINX_LLM_CONSENSUS_K", tv(3, (2, 2, 3, 4, 5)), ConfigDomain.LLM)
    s("JINX_LLM_CONSENSUS_JUDGE", "1", ConfigDomain.LLM)
    s("JINX_LLM_CONSENSUS_JUDGE_MS", tv(450, (300, 350, 450, 600, 800)), ConfigDomain.LLM)
    s("JINX_CODEGRAPH_CTX", "1", ConfigDomain.LLM)

    # ═══════════════════════════════════════════════════════════════
    # AUTOPATCH
    # ═══════════════════════════════════════════════════════════════
    s("JINX_AUTOPATCH_MAX_MS", tv(900, (500, 700, 900, 1200, 1800)), ConfigDomain.LLM)
    s("JINX_AUTOPATCH_PREVIEW_CONC", tv(4, (2, 3, 4, 6, 8)), ConfigDomain.LLM)
    s("JINX_AUTOPATCH_SEARCH_TOPK", tv(4, (2, 3, 4, 6, 8)), ConfigDomain.LLM)
    s("JINX_AUTOPATCH_NO_BUDGETS", "0", ConfigDomain.LLM)
    s("JINX_PATCH_CONTEXT_TOL", "0.72", ConfigDomain.LLM)
    s("JINX_AUTOPATCH_BANDIT_HALF_SEC", tv(1800, (900, 1200, 1800, 2700, 3600)), ConfigDomain.LLM)

    # ═══════════════════════════════════════════════════════════════
    # SELF-REPROGRAM
    # ═══════════════════════════════════════════════════════════════
    s("JINX_PLAN_TOPK", tv(8, (4, 6, 8, 12, 16)), ConfigDomain.LLM)
    s("JINX_PLAN_EMBED_MS", tv(600, (400, 500, 600, 800, 1000)), ConfigDomain.LLM)
    s("JINX_PLAN_REFINE_MS", tv(500, (300, 400, 500, 700, 900)), ConfigDomain.LLM)
    s("JINX_REPROGRAM_TESTS", "1", ConfigDomain.LLM)

    # ═══════════════════════════════════════════════════════════════
    # UI
    # ═══════════════════════════════════════════════════════════════
    s("JINX_SPINNER_ENABLE", "1", ConfigDomain.UI)
    s("JINX_SPINNER_MODE", "toolbar", ConfigDomain.UI)
    s("JINX_SPINNER_MIN_UPDATE_MS", tv(160, (200, 180, 160, 140, 120)), ConfigDomain.UI)
    s("JINX_SPINNER_REDRAW_ONLY_ON_CHANGE", "1", ConfigDomain.UI)
    s("JINX_INCLUDE_SYSTEM_DESC", "1", ConfigDomain.UI)
    s("JINX_LOCALE", "en", ConfigDomain.UI)

    # ═══════════════════════════════════════════════════════════════
    # VALIDATORS & MACROS
    # ═══════════════════════════════════════════════════════════════
    s("JINX_PATCH_AUTOCOMMIT", "1", ConfigDomain.LLM)
    s("JINX_PATCH_CHECK_SYNTAX", "1", ConfigDomain.LLM)
    s("JINX_PATCH_AUTO_INDENT", "1", ConfigDomain.LLM)
    s("JINX_VALIDATORS_ENABLE", "1", ConfigDomain.CORE)
    s("JINX_AUTOMACROS", "1", ConfigDomain.CORE)
    s("JINX_AUTOMACRO_DIALOGUE", "1", ConfigDomain.CORE)
    s("JINX_AUTOMACRO_PROJECT", "1", ConfigDomain.CORE)
    s("JINX_AUTOMACRO_CODE", "1", ConfigDomain.CORE)

    # ═══════════════════════════════════════════════════════════════
    # STATE & PERSISTENCE
    # ═══════════════════════════════════════════════════════════════
    s("JINX_CONTINUITY_ENABLE", "1", ConfigDomain.CORE)
    s("JINX_STATEFRAME_ENABLE", "1", ConfigDomain.CORE)
    s("JINX_CHAINED_PERSIST_BRAIN", "1", ConfigDomain.BRAIN)
    s("JINX_BRAIN_DIR", ".jinx/brain", ConfigDomain.BRAIN)

    # ═══════════════════════════════════════════════════════════════
    # MULTI-SPLIT & PREFETCH
    # ═══════════════════════════════════════════════════════════════
    s("JINX_MULTI_SPLIT_ENABLE", tb(PerformanceTier.MEDIUM), ConfigDomain.RUNTIME)
    s("JINX_MULTI_SPLIT_MAX", tv(6, (3, 4, 6, 8, 12)), ConfigDomain.RUNTIME)
    s("JINX_PREFETCH_BROKER_CONC", tv(3, (1, 2, 3, 4, 6)), ConfigDomain.RUNTIME)

    # ═══════════════════════════════════════════════════════════════
    # VERIFICATION
    # ═══════════════════════════════════════════════════════════════
    s("JINX_VERIFY_AUTORUN", "1", ConfigDomain.CORE)
    s("JINX_AUTOMACRO_VERIFY_EXPORTS", "1", ConfigDomain.CORE)

    cfg._save_state()
    cfg._applied = True

