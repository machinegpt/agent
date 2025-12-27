from __future__ import annotations

import asyncio
import json
import math
import os
import random
import tempfile
import time
from array import array
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Callable, Coroutine, Deque, Dict, Final, List, 
    NamedTuple, Optional, Sequence, Tuple, TypeVar, Union
)

from jinx.log_paths import AUTOTUNE_STATE
from jinx.settings import Settings

T = TypeVar("T")
F = TypeVar("F", bound=float)

_E: Final[float] = math.e
_LN2: Final[float] = math.log(2)
_INV_SQRT_2PI: Final[float] = 1.0 / math.sqrt(2 * math.pi)


class CognitiveMode(IntEnum):
    REACTIVE = 0
    PREDICTIVE = 1
    ADAPTIVE = 2
    DEFENSIVE = 3
    AGGRESSIVE = 4
    DORMANT = 5


class SignalType(IntEnum):
    SATURATION = auto()
    LATENCY = auto()
    ERROR_RATE = auto()
    THROUGHPUT = auto()
    MEMORY_PRESSURE = auto()
    CPU_LOAD = auto()


class DecisionConfidence(NamedTuple):
    action: int
    probability: float
    entropy: float
    evidence: float


@dataclass(slots=True)
class NeuralWeight:
    w: float = 0.0
    momentum: float = 0.0
    velocity: float = 0.0
    
    def update(self, grad: float, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999) -> None:
        self.momentum = beta1 * self.momentum + (1 - beta1) * grad
        self.velocity = beta2 * self.velocity + (1 - beta2) * grad * grad
        self.w -= lr * self.momentum / (math.sqrt(self.velocity) + 1e-8)


class RingBuffer:
    __slots__ = ("_data", "_size", "_idx", "_count")
    
    def __init__(self, size: int) -> None:
        self._data = array("d", [0.0] * size)
        self._size = size
        self._idx = 0
        self._count = 0
    
    def push(self, v: float) -> None:
        self._data[self._idx] = v
        self._idx = (self._idx + 1) % self._size
        self._count = min(self._count + 1, self._size)
    
    def mean(self) -> float:
        return sum(self._data[:self._count]) / max(1, self._count) if self._count else 0.0
    
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        m = self.mean()
        return sum((x - m) ** 2 for x in self._data[:self._count]) / (self._count - 1)
    
    def std(self) -> float:
        return math.sqrt(self.variance())
    
    def trend(self, window: int = 5) -> float:
        if self._count < 2:
            return 0.0
        n = min(window, self._count)
        recent = [self._data[(self._idx - 1 - i) % self._size] for i in range(n)]
        return (recent[0] - recent[-1]) / n if n > 1 else 0.0
    
    def percentile(self, p: float) -> float:
        if not self._count:
            return 0.0
        sorted_data = sorted(self._data[:self._count])
        k = (self._count - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f) if f != c else sorted_data[int(k)]


class BayesianEstimator:
    __slots__ = ("_alpha", "_beta", "_n", "_sum", "_sum_sq")
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self._alpha = prior_alpha
        self._beta = prior_beta
        self._n = 0
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def update(self, x: float, success: bool = True) -> None:
        self._n += 1
        self._sum += x
        self._sum_sq += x * x
        if success:
            self._alpha += 1
        else:
            self._beta += 1
    
    def posterior_mean(self) -> float:
        return self._alpha / (self._alpha + self._beta)
    
    def posterior_variance(self) -> float:
        ab = self._alpha + self._beta
        return (self._alpha * self._beta) / (ab * ab * (ab + 1))
    
    def confidence_interval(self, z: float = 1.96) -> Tuple[float, float]:
        m = self.posterior_mean()
        s = math.sqrt(self.posterior_variance())
        return max(0, m - z * s), min(1, m + z * s)
    
    def entropy(self) -> float:
        p = self.posterior_mean()
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log(p + 1e-10) - (1 - p) * math.log(1 - p + 1e-10)


class KalmanFilter:
    __slots__ = ("_x", "_p", "_q", "_r", "_k")
    
    def __init__(self, initial: float = 0.0, process_noise: float = 0.01, measurement_noise: float = 0.1) -> None:
        self._x = initial
        self._p = 1.0
        self._q = process_noise
        self._r = measurement_noise
        self._k = 0.0
    
    def predict(self, dt: float = 1.0) -> float:
        self._p += self._q * dt
        return self._x
    
    def update(self, z: float) -> float:
        self._k = self._p / (self._p + self._r)
        self._x += self._k * (z - self._x)
        self._p *= (1 - self._k)
        return self._x
    
    @property
    def state(self) -> float:
        return self._x
    
    @property
    def uncertainty(self) -> float:
        return self._p


class AnomalyDetector:
    __slots__ = ("_buffer", "_threshold_z", "_ewma", "_ewmvar", "_alpha")
    
    def __init__(self, window: int = 100, threshold_z: float = 3.0, alpha: float = 0.1) -> None:
        self._buffer = RingBuffer(window)
        self._threshold_z = threshold_z
        self._ewma = 0.0
        self._ewmvar = 0.0
        self._alpha = alpha
    
    def observe(self, x: float) -> Tuple[bool, float]:
        self._buffer.push(x)
        delta = x - self._ewma
        self._ewma += self._alpha * delta
        self._ewmvar = (1 - self._alpha) * (self._ewmvar + self._alpha * delta * delta)
        std = math.sqrt(self._ewmvar) + 1e-10
        z_score = abs(x - self._ewma) / std
        return z_score > self._threshold_z, z_score


class CircuitBreaker:
    __slots__ = ("_failures", "_successes", "_state", "_last_failure", "_cooldown", "_threshold", "_half_open_max")
    
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2
    
    def __init__(self, threshold: int = 5, cooldown: float = 30.0, half_open_max: int = 3) -> None:
        self._failures = 0
        self._successes = 0
        self._state = self.CLOSED
        self._last_failure = 0.0
        self._cooldown = cooldown
        self._threshold = threshold
        self._half_open_max = half_open_max
    
    def record_success(self) -> None:
        if self._state == self.HALF_OPEN:
            self._successes += 1
            if self._successes >= self._half_open_max:
                self._state = self.CLOSED
                self._failures = 0
                self._successes = 0
        elif self._state == self.CLOSED:
            self._failures = max(0, self._failures - 1)
    
    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure = time.monotonic()
        if self._failures >= self._threshold:
            self._state = self.OPEN
    
    def allow(self) -> bool:
        if self._state == self.CLOSED:
            return True
        if self._state == self.OPEN:
            if time.monotonic() - self._last_failure >= self._cooldown:
                self._state = self.HALF_OPEN
                self._successes = 0
                return True
            return False
        return True
    
    @property
    def is_open(self) -> bool:
        return self._state == self.OPEN


class ReinforcementLearner:
    __slots__ = ("_q_table", "_alpha", "_gamma", "_epsilon", "_decay", "_min_epsilon", "_actions", "_last_state", "_last_action")
    
    def __init__(self, n_actions: int = 5, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.3) -> None:
        self._q_table: Dict[int, array] = {}
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._decay = 0.995
        self._min_epsilon = 0.01
        self._actions = n_actions
        self._last_state: Optional[int] = None
        self._last_action: Optional[int] = None
    
    def _discretize(self, state: Tuple[float, ...]) -> int:
        return hash(tuple(int(s * 100) for s in state)) % (2 ** 20)
    
    def _get_q(self, state_id: int) -> array:
        if state_id not in self._q_table:
            self._q_table[state_id] = array("d", [0.0] * self._actions)
        return self._q_table[state_id]
    
    def select_action(self, state: Tuple[float, ...]) -> int:
        state_id = self._discretize(state)
        self._last_state = state_id
        if random.random() < self._epsilon:
            action = random.randint(0, self._actions - 1)
        else:
            q = self._get_q(state_id)
            action = max(range(self._actions), key=lambda a: q[a])
        self._last_action = action
        return action
    
    def update(self, reward: float, next_state: Tuple[float, ...], done: bool = False) -> None:
        if self._last_state is None or self._last_action is None:
            return
        next_id = self._discretize(next_state)
        q = self._get_q(self._last_state)
        next_q = self._get_q(next_id)
        target = reward + (0 if done else self._gamma * max(next_q))
        q[self._last_action] += self._alpha * (target - q[self._last_action])
        self._epsilon = max(self._min_epsilon, self._epsilon * self._decay)


class PredictiveModel:
    __slots__ = ("_weights", "_bias", "_history", "_horizon")
    
    def __init__(self, input_dim: int = 10, horizon: int = 5) -> None:
        self._weights = [NeuralWeight(random.gauss(0, 0.1)) for _ in range(input_dim)]
        self._bias = NeuralWeight(0.0)
        self._history: Deque[float] = deque(maxlen=input_dim)
        self._horizon = horizon
    
    def observe(self, x: float) -> None:
        self._history.append(x)
    
    def predict(self) -> float:
        if len(self._history) < len(self._weights):
            return self._history[-1] if self._history else 0.0
        inputs = list(self._history)[-len(self._weights):]
        return sum(w.w * x for w, x in zip(self._weights, inputs)) + self._bias.w
    
    def train(self, target: float, lr: float = 0.01) -> float:
        if len(self._history) < len(self._weights):
            return 0.0
        inputs = list(self._history)[-len(self._weights):]
        pred = self.predict()
        error = target - pred
        for w, x in zip(self._weights, inputs):
            w.update(-error * x, lr)
        self._bias.update(-error, lr)
        return error * error


class CognitiveCortex:
    __slots__ = (
        "_mode", "_signals", "_kalman_filters", "_anomaly_detectors", "_circuit_breakers",
        "_bayesian", "_rl_agent", "_predictive", "_decision_history", "_reward_history",
        "_mode_durations", "_last_mode_change", "_confidence_threshold", "_reaction_time_ns"
    )
    
    def __init__(self) -> None:
        self._mode = CognitiveMode.REACTIVE
        self._signals: Dict[SignalType, RingBuffer] = {s: RingBuffer(256) for s in SignalType}
        self._kalman_filters: Dict[SignalType, KalmanFilter] = {s: KalmanFilter() for s in SignalType}
        self._anomaly_detectors: Dict[SignalType, AnomalyDetector] = {s: AnomalyDetector() for s in SignalType}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._bayesian = BayesianEstimator()
        self._rl_agent = ReinforcementLearner(n_actions=len(CognitiveMode))
        self._predictive = PredictiveModel()
        self._decision_history: Deque[DecisionConfidence] = deque(maxlen=100)
        self._reward_history = RingBuffer(100)
        self._mode_durations: Dict[CognitiveMode, float] = {m: 0.0 for m in CognitiveMode}
        self._last_mode_change = time.monotonic()
        self._confidence_threshold = 0.6
        self._reaction_time_ns = 0
    
    def _compute_state_vector(self) -> Tuple[float, ...]:
        return tuple(
            self._kalman_filters[s].state for s in SignalType
        ) + (
            self._bayesian.posterior_mean(),
            self._bayesian.entropy(),
            float(self._mode) / len(CognitiveMode),
        )
    
    def _compute_reward(self, metrics: "CortexMetrics") -> float:
        throughput_reward = min(1.0, metrics.throughput / max(1, metrics.target_throughput))
        latency_penalty = max(0, (metrics.avg_latency_ms - metrics.target_latency_ms) / metrics.target_latency_ms)
        error_penalty = metrics.error_rate * 2
        stability_bonus = 1.0 / (1.0 + metrics.mode_switches)
        return throughput_reward - latency_penalty - error_penalty + stability_bonus * 0.1
    
    def ingest(self, signal: SignalType, value: float) -> Tuple[bool, float]:
        t0 = time.perf_counter_ns()
        self._signals[signal].push(value)
        filtered = self._kalman_filters[signal].update(value)
        is_anomaly, z_score = self._anomaly_detectors[signal].observe(value)
        self._predictive.observe(filtered)
        self._reaction_time_ns = time.perf_counter_ns() - t0
        return is_anomaly, z_score
    
    def decide(self, metrics: "CortexMetrics") -> CognitiveMode:
        state = self._compute_state_vector()
        action = self._rl_agent.select_action(state)
        new_mode = CognitiveMode(action % len(CognitiveMode))
        
        sat_pred = self._predictive.predict()
        sat_actual = self._signals[SignalType.SATURATION].mean()
        
        anomalies = sum(
            1 for s in SignalType 
            if self._anomaly_detectors[s].observe(self._signals[s].mean())[0]
        )
        
        if anomalies >= 2:
            new_mode = CognitiveMode.DEFENSIVE
        elif sat_pred > 0.8 and sat_actual > 0.6:
            new_mode = CognitiveMode.AGGRESSIVE
        elif sat_pred < 0.2 and sat_actual < 0.3:
            new_mode = CognitiveMode.DORMANT
        
        confidence = DecisionConfidence(
            action=new_mode,
            probability=self._bayesian.posterior_mean(),
            entropy=self._bayesian.entropy(),
            evidence=1.0 - (anomalies / len(SignalType))
        )
        self._decision_history.append(confidence)
        
        if new_mode != self._mode:
            now = time.monotonic()
            self._mode_durations[self._mode] += now - self._last_mode_change
            self._last_mode_change = now
            self._mode = new_mode
        
        reward = self._compute_reward(metrics)
        self._reward_history.push(reward)
        self._rl_agent.update(reward, state, done=False)
        self._bayesian.update(sat_actual, success=reward > 0)
        self._predictive.train(sat_actual)
        
        return self._mode
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker()
        return self._circuit_breakers[name]


@dataclass(slots=True)
class CortexMetrics:
    throughput: float = 0.0
    target_throughput: float = 100.0
    avg_latency_ms: float = 0.0
    target_latency_ms: float = 50.0
    error_rate: float = 0.0
    mode_switches: int = 0
    saturation: float = 0.0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0


class PersistentMemory:
    __slots__ = ("_path", "_cache", "_dirty", "_last_sync")
    
    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: Dict[str, Any] = {}
        self._dirty = False
        self._last_sync = 0.0
    
    def load(self) -> None:
        if self._path.exists():
            with self._path.open("r") as f:
                self._cache = json.load(f)
    
    def save(self, force: bool = False) -> None:
        if not self._dirty and not force:
            return
        if time.monotonic() - self._last_sync < 1.0 and not force:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent))
        with os.fdopen(fd, "w") as f:
            json.dump(self._cache, f)
        os.replace(tmp, str(self._path))
        self._dirty = False
        self._last_sync = time.monotonic()
    
    def get(self, key: str, default: T = None) -> T:
        return self._cache.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._dirty = True


class CognitiveEngine:
    __slots__ = (
        "_queue", "_settings", "_cortex", "_memory", "_metrics", "_running",
        "_tick_count", "_last_tick", "_tick_budget_ns", "_adaptation_rate",
        "_emergency_mode", "_health_score"
    )
    
    def __init__(self, queue: asyncio.Queue[str], settings: Settings) -> None:
        self._queue = queue
        self._settings = settings
        self._cortex = CognitiveCortex()
        self._memory = PersistentMemory(Path(AUTOTUNE_STATE))
        self._metrics = CortexMetrics()
        self._running = True
        self._tick_count = 0
        self._last_tick = time.monotonic()
        self._tick_budget_ns = settings.runtime.hard_rt_budget_ms * 1_000_000
        self._adaptation_rate = 0.1
        self._emergency_mode = False
        self._health_score = 1.0
    
    def _restore(self) -> None:
        self._memory.load()
        rt = self._settings.runtime
        rt.use_priority_queue = self._memory.get("priority_queue", rt.use_priority_queue)
        rt.hard_rt_budget_ms = self._memory.get("budget_ms", rt.hard_rt_budget_ms)
        self._health_score = self._memory.get("health_score", 1.0)
    
    def _persist(self) -> None:
        rt = self._settings.runtime
        self._memory.set("priority_queue", rt.use_priority_queue)
        self._memory.set("budget_ms", rt.hard_rt_budget_ms)
        self._memory.set("health_score", self._health_score)
        self._memory.set("mode", int(self._cortex._mode))
        self._memory.set("tick_count", self._tick_count)
        self._memory.save()
    
    def _sample_signals(self) -> None:
        maxsize = getattr(self._queue, "maxsize", 1) or 1
        saturation = self._queue.qsize() / maxsize
        self._cortex.ingest(SignalType.SATURATION, saturation)
        self._cortex.ingest(SignalType.THROUGHPUT, self._metrics.throughput)
        self._cortex.ingest(SignalType.ERROR_RATE, self._metrics.error_rate)
        self._cortex.ingest(SignalType.LATENCY, self._metrics.avg_latency_ms / 1000)
        self._metrics.saturation = saturation
    
    def _apply_mode(self, mode: CognitiveMode) -> None:
        rt = self._settings.runtime
        
        match mode:
            case CognitiveMode.REACTIVE:
                rt.use_priority_queue = False
                rt.hard_rt_budget_ms = 40
            case CognitiveMode.PREDICTIVE:
                rt.use_priority_queue = True
                rt.hard_rt_budget_ms = 35
            case CognitiveMode.ADAPTIVE:
                rt.use_priority_queue = True
                rt.hard_rt_budget_ms = 30
            case CognitiveMode.DEFENSIVE:
                rt.use_priority_queue = True
                rt.hard_rt_budget_ms = 20
                self._emergency_mode = True
            case CognitiveMode.AGGRESSIVE:
                rt.use_priority_queue = True
                rt.hard_rt_budget_ms = 15
            case CognitiveMode.DORMANT:
                rt.use_priority_queue = False
                rt.hard_rt_budget_ms = 50
        
        if mode != CognitiveMode.DEFENSIVE:
            self._emergency_mode = False
    
    def _compute_health(self) -> float:
        error_factor = 1.0 - self._metrics.error_rate
        latency_factor = 1.0 / (1.0 + max(0, self._metrics.avg_latency_ms - self._metrics.target_latency_ms) / 100)
        saturation_factor = 1.0 - self._metrics.saturation * 0.5
        stability = 1.0 / (1.0 + self._metrics.mode_switches * 0.1)
        return (error_factor * 0.3 + latency_factor * 0.3 + saturation_factor * 0.2 + stability * 0.2)
    
    async def _tick(self) -> None:
        t0 = time.perf_counter_ns()
        
        self._sample_signals()
        mode = self._cortex.decide(self._metrics)
        self._apply_mode(mode)
        self._health_score = 0.9 * self._health_score + 0.1 * self._compute_health()
        
        elapsed_ns = time.perf_counter_ns() - t0
        if elapsed_ns > self._tick_budget_ns:
            self._adaptation_rate *= 0.9
        else:
            self._adaptation_rate = min(1.0, self._adaptation_rate * 1.01)
        
        self._tick_count += 1
        if self._tick_count % 100 == 0:
            self._persist()
    
    async def run(self) -> None:
        self._restore()
        
        base_interval = self._settings.runtime.saturate_window_ms / 1000.0
        
        while self._running:
            await self._tick()
            
            interval = base_interval
            if self._emergency_mode:
                interval *= 0.5
            elif self._cortex._mode == CognitiveMode.DORMANT:
                interval *= 2.0
            
            interval *= (2.0 - self._adaptation_rate)
            await asyncio.sleep(max(0.01, interval))
    
    def stop(self) -> None:
        self._running = False
        self._persist()


def start_autotune_task(q_in: asyncio.Queue[str], settings: Settings) -> asyncio.Task[None]:
    engine = CognitiveEngine(q_in, settings)
    return asyncio.create_task(engine.run(), name="cognitive-engine")
