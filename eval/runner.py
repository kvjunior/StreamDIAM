"""Streaming run assembly (C5).

Wires the store, a view maintainer, the drift monitor+trigger, and the
background trainer+hot-swap into one prequential run, and provides the retrain
hook that keeps retraining decoupled from the loop. A run emits results only --
``RunResult`` and its JSONL records -- never plots; figures read the committed
records in a separate step, which is the anti-fabrication discipline (figures can
never invent data they are not given).

``run_streaming`` is maintainer-agnostic via ``maintainer_factory`` (default
from-scratch); later batches inject the incremental maintainer and the baselines
to run every strategy through the identical harness.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from streamdiam.data.schema import TransactionBatch
from streamdiam.data.stream_replay import replay_by_time
from streamdiam.drift import DriftMonitor, PeriodicTrigger, RetrainTrigger, Trigger
from streamdiam.eval.protocol import (
    FromScratchStreamingMaintainer,
    PrequentialProtocol,
    RunResult,
    StreamingMaintainer,
)
from streamdiam.model.diam import DIAM
from streamdiam.retrain import BackgroundTrainer, HotSwapRegistry, TrainConfig
from streamdiam.store.interface import TemporalMultigraphStore
from streamdiam.store.reference_py import ReferencePyStore

__all__ = ["RunConfig", "run_streaming", "write_run_jsonl"]


@dataclass
class RunConfig:
    """Everything that parameterises one streaming run.

    ``maintainer`` selects the view strategy (``from_scratch`` | ``incremental``
    | ``frozen``). ``retrain_policy`` selects when weights update (``drift`` =
    drift-triggered, our method; ``periodic`` = full-retrain every
    ``retrain_period`` ticks; ``never`` = frozen weights after warmup).
    """

    threshold: float = 0.5
    t_max: int = 10_000
    chunk_size: int = 10_000
    warmup_fraction: float = 0.0  # fraction of distinct timesteps used to pre-train
    maintainer: str = "from_scratch"
    retrain_policy: str = "drift"
    retrain_period: int = 10
    cooldown: int = 5
    train_config: TrainConfig = field(default_factory=TrainConfig)


MaintainerFactory = Callable[
    [Callable[[], DIAM], TemporalMultigraphStore, int], StreamingMaintainer
]

def _resolve_maintainer(name: str) -> MaintainerFactory:
    """Look up a maintainer preset by name.

    Resolved lazily (and the incremental/frozen strategies imported here rather
    than at module top) to avoid an import cycle: ``baselines.maintainers``
    imports this package's ``eval.protocol``.
    """
    if name == "from_scratch":
        return FromScratchStreamingMaintainer
    if name in ("incremental", "frozen"):
        from streamdiam.baselines.maintainers import (
            FrozenViewMaintainer,
            IncrementalStreamingMaintainer,
        )

        return (
            IncrementalStreamingMaintainer
            if name == "incremental"
            else FrozenViewMaintainer
        )
    raise ValueError(f"unknown maintainer preset: {name!r}")


def run_streaming(
    edges: TransactionBatch,
    labels: npt.NDArray[np.integer],
    universe_size: int,
    model: DIAM,
    config: RunConfig | None = None,
    *,
    maintainer_factory: MaintainerFactory | None = None,
    enable_retrain: bool = True,
) -> RunResult:
    """Assemble and run one prequential streaming evaluation.

    Args:
        edges: the full canonical-order transaction stream.
        labels: ``[universe_size]`` ground truth in ``{-1, 0, 1}``.
        universe_size: account-id space size.
        model: the detector; used as the initial live version (fine-tuned in
            place of a warmup only if ``warmup_fraction > 0``).
        config: run parameters.
        maintainer_factory: view-maintenance strategy (default from-scratch).
        enable_retrain: if False, no background retraining (frozen-after-warmup).
    """
    cfg = config or RunConfig()
    factory = maintainer_factory or _resolve_maintainer(cfg.maintainer)
    n = int(universe_size)
    labels = np.asarray(labels)

    store: TemporalMultigraphStore = ReferencePyStore(model.in_dim)
    registry = HotSwapRegistry(model)
    trainer = BackgroundTrainer(t_max=cfg.t_max)

    # Optional warmup: pre-train on an initial time-prefix so the first streamed
    # tick is scored by a trained model. Warmup accounts are marked seen so they
    # never count as leakage-free first appearances (they were trained on).
    initial_seen = np.zeros(n, dtype=bool)
    ts = np.asarray(edges.timestamp)
    distinct = np.unique(ts)
    stream_edges = edges
    if cfg.warmup_fraction > 0.0 and distinct.size > 1:
        cut = max(1, int(distinct.size * cfg.warmup_fraction))
        t_cut = int(distinct[cut - 1])
        split = int(np.searchsorted(ts, t_cut, side="right"))
        warm = edges[:split]
        assert isinstance(warm, TransactionBatch)
        stream_rest = edges[split:]
        assert isinstance(stream_rest, TransactionBatch)
        stream_edges = stream_rest
        if len(warm) > 0:
            store.insert(warm)
            warm_acc = np.unique(np.concatenate([warm.src, warm.dst]))
            initial_seen[warm_acc] = True
            ckpt = trainer.train(
                registry.model, store, labels, n, float(t_cut), cfg.train_config
            )
            registry.swap(ckpt)

    maintainer = factory(lambda: registry.model, store, n)

    # Retrain policy: drift-triggered (ours), periodic full-retrain, or never.
    monitor: DriftMonitor | None = None
    trigger: Trigger | None = None
    if enable_retrain and cfg.retrain_policy == "drift":
        monitor = DriftMonitor()
        trigger = RetrainTrigger(monitor, cooldown=cfg.cooldown)
    elif enable_retrain and cfg.retrain_policy == "periodic":
        trigger = PeriodicTrigger(period=cfg.retrain_period)
    elif enable_retrain and cfg.retrain_policy != "never":
        raise ValueError(f"unknown retrain_policy: {cfg.retrain_policy!r}")

    def retrain_hook(ref_time: int, seen_mask: npt.NDArray[np.bool_]) -> int | None:
        if not enable_retrain:
            return None
        try:
            ckpt = trainer.train(
                registry.model, store, labels, n, float(ref_time),
                cfg.train_config, train_mask=seen_mask,
            )
        except ValueError:
            return None  # no labelled accounts in the window yet
        registry.swap(ckpt)
        return registry.version

    protocol = PrequentialProtocol(
        store,
        n,
        maintainer,
        labels,
        monitor=monitor,
        trigger=trigger,
        retrain_hook=retrain_hook,
        version_provider=lambda: registry.version,
        threshold=cfg.threshold,
        initial_seen=initial_seen,
    )
    return protocol.run(replay_by_time(stream_edges))


def write_run_jsonl(result: RunResult, path: str | Path) -> Path:
    """Write a run as JSONL: one summary record, then one record per tick.

    This is the committed artifact figures read from; nothing here plots.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        fh.write(json.dumps({"record": "summary", **result.summary_dict()}) + "\n")
        for tick in result.ticks:
            fh.write(json.dumps({"record": "tick", **tick.as_dict()}) + "\n")
    return p
