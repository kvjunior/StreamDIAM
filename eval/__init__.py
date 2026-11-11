"""Rigorous streaming evaluation (C5): one honest prequential harness.

Every streaming number the paper reports comes from this harness -- metrics,
the test-then-train protocol, and the runner that wires store + maintainer +
drift + retrain into a run and emits results only. One harness, so the
incremental-vs-retrain comparison is internally consistent (resolves R2 O7).
"""

from streamdiam.eval.metrics import (
    PrequentialAccumulator,
    WindowMetrics,
    compute_classification_metrics,
)
from streamdiam.eval.protocol import (
    FromScratchStreamingMaintainer,
    PrequentialProtocol,
    RunResult,
    StreamingMaintainer,
    TickRecord,
)
from streamdiam.eval.runner import RunConfig, run_streaming, write_run_jsonl

__all__ = [
    "WindowMetrics",
    "PrequentialAccumulator",
    "compute_classification_metrics",
    "StreamingMaintainer",
    "FromScratchStreamingMaintainer",
    "PrequentialProtocol",
    "TickRecord",
    "RunResult",
    "RunConfig",
    "run_streaming",
    "write_run_jsonl",
]
