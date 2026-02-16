"""Lightweight audit + SLO counters and resilience failpoint hooks."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class SLOMetric:
    count: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0


_metrics: dict[str, SLOMetric] = defaultdict(SLOMetric)
_audit_events: list[dict[str, Any]] = []


def record_metric(name: str, latency_ms: float, success: bool) -> None:
    m = _metrics[name]
    m.count += 1
    if not success:
        m.failures += 1
    m.total_latency_ms += latency_ms


def get_metric_snapshot() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, m in _metrics.items():
        avg = (m.total_latency_ms / m.count) if m.count else 0.0
        out[key] = {
            "count": float(m.count),
            "failures": float(m.failures),
            "avg_latency_ms": avg,
            "error_rate": (m.failures / m.count) if m.count else 0.0,
        }
    return out


def emit_audit_event(event_type: str, **payload: Any) -> None:
    _audit_events.append({"ts": time.time(), "event": event_type, **payload})
    max_events = int(os.getenv("AUDIT_EVENT_BUFFER", "1000"))
    if len(_audit_events) > max_events:
        del _audit_events[0 : len(_audit_events) - max_events]


def get_audit_events(limit: int = 100) -> list[dict[str, Any]]:
    return _audit_events[-limit:]


def should_inject_failure(operation: str) -> bool:
    failpoints = {x.strip() for x in os.getenv("RESILIENCE_FAILPOINTS", "").split(",") if x.strip()}
    return operation in failpoints
