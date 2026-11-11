"""Decoupled drift-triggered retraining (C4): recency, background trainer, hot-swap.

Off the critical path: when :mod:`streamdiam.drift` fires, the background trainer
fine-tunes a copy of the model on a recency-weighted labelled window and the
hot-swap registry installs the result atomically. The live inference view keeps
serving current scores throughout.
"""

from streamdiam.retrain.background_trainer import (
    BackgroundTrainer,
    ModelCheckpoint,
    TrainConfig,
)
from streamdiam.retrain.hot_swap import HotSwapRegistry, SwapEvent, VersionedModel
from streamdiam.retrain.recency_weight import RecencyWeighting, recency_weights

__all__ = [
    "BackgroundTrainer",
    "ModelCheckpoint",
    "TrainConfig",
    "HotSwapRegistry",
    "SwapEvent",
    "VersionedModel",
    "RecencyWeighting",
    "recency_weights",
]
