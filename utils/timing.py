"""Honest wall-clock timing for the streaming harness.

Latency and throughput are first-class results (RQ1), so the harness measures
them with a monotonic clock and reports the tail, not just the mean -- a p99
update latency is what a real-time system lives or dies by. No CUDA-event timing
here (the reference runs on CPU); a GPU timer can wrap the same interface later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Stopwatch", "LatencyRecorder", "LatencySummary"]


class Stopwatch:
    """Context manager timing one block with :func:`time.perf_counter`.

    ``with Stopwatch() as sw: ...`` then read ``sw.elapsed`` (seconds).
    """

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._t0 = 0.0

    def __enter__(self) -> Stopwatch:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0


@dataclass(frozen=True)
class LatencySummary:
    """Summary statistics over a set of latency samples (all in milliseconds)."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    total_s: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
            "total_s": self.total_s,
        }


@dataclass
class LatencyRecorder:
    """Collects per-event latency samples and summarises the distribution."""

    samples_s: list[float] = field(default_factory=list)

    def record(self, seconds: float) -> None:
        self.samples_s.append(float(seconds))

    def __len__(self) -> int:
        return len(self.samples_s)

    @property
    def throughput_eps(self) -> float:
        """Events per second over the recorded samples (0 if none)."""
        total = sum(self.samples_s)
        return len(self.samples_s) / total if total > 0 else 0.0

    def summary(self) -> LatencySummary:
        if not self.samples_s:
            return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        a = np.asarray(self.samples_s, dtype=np.float64) * 1000.0  # -> ms
        return LatencySummary(
            count=a.size,
            mean_ms=float(a.mean()),
            p50_ms=float(np.percentile(a, 50)),
            p95_ms=float(np.percentile(a, 95)),
            p99_ms=float(np.percentile(a, 99)),
            max_ms=float(a.max()),
            total_s=float(a.sum() / 1000.0),
        )
