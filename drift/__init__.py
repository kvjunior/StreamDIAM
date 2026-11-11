"""Concept-drift detection (C4): detectors, monitor, retrain trigger.

The critical path stays inference-only; this package watches the stream and
decides *when* the decoupled background retrainer should run. Its firing log is
the empirical heart of the paper's reframing (RQ3: how rarely retraining is
actually needed).
"""

from streamdiam.drift.detectors import (
    ADWIN,
    DDM,
    DriftDetector,
    DriftStatus,
    PageHinkley,
)
from streamdiam.drift.monitor import DriftMonitor, PrequentialError
from streamdiam.drift.trigger import (
    PeriodicTrigger,
    RetrainSignal,
    RetrainTrigger,
    Trigger,
    TriggerEvent,
)

__all__ = [
    "ADWIN",
    "DDM",
    "DriftDetector",
    "DriftStatus",
    "PageHinkley",
    "DriftMonitor",
    "PrequentialError",
    "RetrainSignal",
    "RetrainTrigger",
    "PeriodicTrigger",
    "Trigger",
    "TriggerEvent",
]
