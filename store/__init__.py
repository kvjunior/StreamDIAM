"""Temporal multigraph store (C2): interface + backends."""

from streamdiam.store.interface import TemporalMultigraphStore
from streamdiam.store.reference_py import ReferencePyStore

__all__ = ["TemporalMultigraphStore", "ReferencePyStore"]
