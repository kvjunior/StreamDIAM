"""Baselines and maintenance strategies for the streaming comparison (C5).

Maintenance strategies (all run through the one harness): the proposed
incremental maintainer (C3), the frozen-view ablation, and the from-scratch
reference (in ``eval.protocol``). Retrain policies are configured on the runner
(drift-triggered vs. periodic full-retrain vs. never).

External-model baselines (XGB-Graph, TGN, APAN, Spade) are wired through the same
``StreamingMaintainer`` interface in a dedicated step with their pinned upstream
code under ``third_party/`` -- never mocked, which is the integrity line the old
submission failed.
"""

from streamdiam.baselines.maintainers import (
    FrozenViewMaintainer,
    IncrementalStreamingMaintainer,
)
from streamdiam.baselines.xgb_graph import XGBGraphMaintainer, run_xgb_graph

__all__ = [
    "IncrementalStreamingMaintainer",
    "FrozenViewMaintainer",
    "XGBGraphMaintainer",
    "run_xgb_graph",
]
