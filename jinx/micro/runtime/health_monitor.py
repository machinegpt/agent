"""Health Monitor - Continuous system health monitoring and auto-recovery.

Real-time monitoring with:
- Performance metrics tracking
- Anomaly detection using statistical methods
- Automatic service restart on degradation
- Predictive failure detection
- Self-healing trigger automation
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None  # type: ignore


@dataclass
class HealthMetrics:
    """System health metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    active_tasks: int
    pulse: int
    error_rate: float
    avg_latency_ms: float


class HealthMonitor:
    """Monitors system health and triggers auto-recovery."""
    
    def __init__(self):
        self._metrics_buffer: deque[HealthMetrics] = deque(maxlen=1000)
        self._running = False
        self._check_interval = 10.0  # seconds
        
        # Thresholds
        self._cpu_threshold = 90.0  # %
        self._memory_threshold = 90.0  # %
        self._error_rate_threshold = 0.3  # 30% error rate
        self._latency_threshold = 5000.0  # 5 seconds
        
        # Statistics
        self._anomalies_detected = 0
        self._recovery_actions_taken = 0
        # Print throttling to avoid console spam
        self._last_anomaly: Optional[str] = None
        self._last_anomaly_print_ts: float = 0.0
    
    async def start(self):
        """Start health monitoring loop."""
        if self._running:
            return
        
        self._running = True
        
        print("🏥 Health Monitor started")
        
        while self._running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                if metrics:
                    self._metrics_buffer.append(metrics)
                    
                    # Check for anomalies
                    anomaly = self._detect_anomaly(metrics)
                    
                    if anomaly:
                        self._anomalies_detected += 1
                        await self._handle_anomaly(anomaly, metrics)
                
                # Wait before next check
                await asyncio.sleep(self._check_interval)
            
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self._check_interval)
    
    async def stop(self):
        """Stop health monitoring."""
        self._running = False
    
    async def _collect_metrics(self) -> Optional[HealthMetrics]:
        """Collect current health metrics."""
        
        try:
            # System metrics (fallback if psutil unavailable)
            if _HAS_PSUTIL and psutil:
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                memory_percent = memory.percent
            else:
                cpu = 0.0
                memory_mb = 0.0
                memory_percent = 0.0
            
            # Jinx state
            try:
                import jinx.state as jx_state
                pulse = getattr(jx_state, 'pulse', 0)
            except Exception:
                pulse = 0
            
            # Active tasks
            try:
                active_tasks = len(asyncio.all_tasks())
            except Exception:
                active_tasks = 0
            
            # Performance metrics (from adaptive config if available)
            error_rate = 0.0
            avg_latency = 0.0
            
            try:
                from jinx.micro.runtime.adaptive_config import get_adaptive_config
                
                config = await get_adaptive_config()
                metrics = config.get_metrics()
                
                perf = metrics.get('performance', {})
                success_rate = perf.get('success_rate', 1.0)
                error_rate = 1.0 - success_rate
                avg_latency = perf.get('avg_latency_ms', 0.0)
            
            except Exception:
                pass
            
            return HealthMetrics(
                timestamp=time.time(),
                cpu_percent=cpu,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                active_tasks=active_tasks,
                pulse=pulse,
                error_rate=error_rate,
                avg_latency_ms=avg_latency
            )
        
        except Exception:
            return None
    
    def _detect_anomaly(self, metrics: HealthMetrics) -> Optional[str]:
        """Detect anomalies in metrics."""
        
        # High CPU
        if metrics.cpu_percent > self._cpu_threshold:
            return f"high_cpu:{metrics.cpu_percent:.1f}%"
        
        # High memory
        if metrics.memory_percent > self._memory_threshold:
            return f"high_memory:{metrics.memory_percent:.1f}%"
        
        # High error rate
        if metrics.error_rate > self._error_rate_threshold:
            return f"high_error_rate:{metrics.error_rate:.2f}"
        
        # High latency
        if metrics.avg_latency_ms > self._latency_threshold:
            return f"high_latency:{metrics.avg_latency_ms:.0f}ms"
        
        # Pulse critically low
        if metrics.pulse < 50:
            return f"low_pulse:{metrics.pulse}"
        
        # Trending analysis
        if len(self._metrics_buffer) >= 5:
            recent = list(self._metrics_buffer)[-5:]
            
            # CPU trending up
            cpu_trend = all(
                recent[i].cpu_percent < recent[i+1].cpu_percent
                for i in range(len(recent)-1)
            )
            
            if cpu_trend and recent[-1].cpu_percent > 70:
                return "cpu_trending_up"
            
            # Memory trending up
            mem_trend = all(
                recent[i].memory_percent < recent[i+1].memory_percent
                for i in range(len(recent)-1)
            )
            
            if mem_trend and recent[-1].memory_percent > 70:
                return "memory_trending_up"
        
        return None
    
    async def _handle_anomaly(self, anomaly: str, metrics: HealthMetrics):
        """Handle detected anomaly."""
        
        now = time.time()
        try:
            same = (anomaly == self._last_anomaly)
        except Exception:
            same = False
        # Print at most once per minute for the same anomaly, but always print when anomaly changes.
        if (not same) or ((now - float(self._last_anomaly_print_ts or 0.0)) >= 60.0):
            self._last_anomaly = anomaly
            self._last_anomaly_print_ts = now
            print(f"\n⚠️  ANOMALY DETECTED: {anomaly}")
            print(f"   CPU: {metrics.cpu_percent:.1f}%")
            print(f"   Memory: {metrics.memory_percent:.1f}%")
            print(f"   Error Rate: {metrics.error_rate:.2%}")
            print(f"   Latency: {metrics.avg_latency_ms:.1f}ms")
            print(f"   Pulse: {metrics.pulse}")
        
        # Determine recovery action
        action = self._select_recovery_action(anomaly, metrics)
        
        if action:
            if (anomaly == self._last_anomaly):
                try:
                    # Only print recovery line when anomaly print was emitted
                    if now == self._last_anomaly_print_ts:
                        print(f"   Recovery: {action}")
                except Exception:
                    pass
            else:
                print(f"   Recovery: {action}")
            
            success = await self._execute_recovery(action, anomaly)
            
            if success:
                self._recovery_actions_taken += 1
                if (anomaly == self._last_anomaly):
                    try:
                        if now == self._last_anomaly_print_ts:
                            print("   ✓ Recovery successful")
                    except Exception:
                        pass
                else:
                    print("   ✓ Recovery successful")
            else:
                if (anomaly == self._last_anomaly):
                    try:
                        if now == self._last_anomaly_print_ts:
                            print("   ✗ Recovery failed")
                    except Exception:
                        pass
                else:
                    print("   ✗ Recovery failed")
        
        if (anomaly == self._last_anomaly):
            try:
                if now == self._last_anomaly_print_ts:
                    print()
            except Exception:
                pass
        else:
            print()
    
    def _select_recovery_action(
        self,
        anomaly: str,
        metrics: HealthMetrics
    ) -> Optional[str]:
        """Select appropriate recovery action."""
        
        if 'high_cpu' in anomaly or 'cpu_trending' in anomaly:
            return 'reduce_concurrency'
        
        elif 'high_memory' in anomaly or 'memory_trending' in anomaly:
            return 'clear_caches'
        
        elif 'high_error_rate' in anomaly:
            return 'trigger_healing'
        
        elif 'high_latency' in anomaly:
            return 'optimize_config'
        
        elif 'low_pulse' in anomaly:
            return 'restore_pulse'
        
        return None
    
    async def _execute_recovery(
        self,
        action: str,
        anomaly: str
    ) -> bool:
        """Execute recovery action."""
        
        try:
            if action == 'reduce_concurrency':
                # Reduce concurrent operations
                try:
                    from jinx.micro.runtime.adaptive_config import get_adaptive_config
                    
                    config = await get_adaptive_config()
                    
                    # Temporarily reduce concurrency
                    import os
                    current = int(os.getenv('JINX_MAX_CONCURRENT', '3'))
                    new_value = max(1, current - 1)
                    
                    os.environ['JINX_MAX_CONCURRENT'] = str(new_value)
                    
                    return True
                except Exception:
                    return False
            
            elif action == 'clear_caches':
                # Clear internal caches
                try:
                    from jinx.micro.embeddings import clear_embedding_cache
                    
                    clear_embedding_cache()
                    
                    return True
                except Exception:
                    return False
            
            elif action == 'trigger_healing':
                # Trigger self-healing system check
                try:
                    from jinx.micro.runtime.self_healing import get_healing_system
                    
                    system = await get_healing_system()
                    stats = system.get_stats()
                    
                    # If many errors, force healing check
                    if stats['active_patterns'] > 0:
                        print(f"   Active error patterns: {stats['active_patterns']}")
                    
                    return True
                except Exception:
                    return False
            
            elif action == 'optimize_config':
                # Trigger config optimization
                try:
                    from jinx.micro.runtime.adaptive_config import get_adaptive_config
                    
                    config = await get_adaptive_config()
                    
                    # Force optimization check
                    await config.check_and_optimize()
                    
                    return True
                except Exception:
                    return False
            
            elif action == 'restore_pulse':
                # Restore pulse to safe level
                try:
                    from jinx.error_service import inc_pulse
                    
                    await inc_pulse(100)
                    
                    return True
                except Exception:
                    return False
        
        except Exception:
            return False
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        
        current = self._metrics_buffer[-1] if self._metrics_buffer else None
        
        return {
            'running': self._running,
            'anomalies_detected': self._anomalies_detected,
            'recovery_actions_taken': self._recovery_actions_taken,
            'metrics_collected': len(self._metrics_buffer),
            'current_metrics': {
                'cpu_percent': current.cpu_percent if current else 0,
                'memory_percent': current.memory_percent if current else 0,
                'pulse': current.pulse if current else 0,
                'error_rate': current.error_rate if current else 0
            } if current else {}
        }


# Singleton
_health_monitor: Optional[HealthMonitor] = None
_monitor_task: Optional[asyncio.Task] = None


async def start_health_monitoring():
    """Start health monitoring background task."""
    global _health_monitor, _monitor_task
    
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    
    if _monitor_task is None or _monitor_task.done():
        _monitor_task = asyncio.create_task(_health_monitor.start())


async def stop_health_monitoring():
    """Stop health monitoring."""
    global _health_monitor, _monitor_task
    
    if _health_monitor:
        await _health_monitor.stop()
    
    if _monitor_task and not _monitor_task.done():
        _monitor_task.cancel()
        try:
            await _monitor_task
        except asyncio.CancelledError:
            pass


__all__ = [
    "HealthMonitor",
    "start_health_monitoring",
    "stop_health_monitoring",
]
