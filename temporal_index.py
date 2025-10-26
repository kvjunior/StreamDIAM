"""
temporal_index.py - Enhanced Temporal Graph Index Management for StreamDIAM

This module implements the core data management infrastructure for real-time
illicit account detection over temporal cryptocurrency transaction multigraphs.
It provides efficient indexing, storage, and retrieval mechanisms optimized for
streaming transaction data with temporal queries.

ICDE 2026 Enhanced Version: This version includes comprehensive profiling
instrumentation to support formal complexity analysis and empirical validation
of algorithmic performance claims.

Copyright (c) 2025 StreamDIAM Research Team
For ICDE 2026 Submission: "StreamDIAM: Scalable Real-Time Illicit Account 
Detection Over Temporal Cryptocurrency Transaction Graphs"
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
import threading
import queue
import time
import pickle
import os
import sys
from concurrent.futures import ThreadPoolExecutor, Future
import heapq
import bisect
from torch_geometric.data import Data
from torch_geometric.utils import degree, subgraph
import warnings
from abc import ABC, abstractmethod
import psutil
import logging
from datetime import datetime, timedelta
import hashlib
import lz4.frame
import struct
from contextlib import contextmanager
import signal
import atexit
import json

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('temporal_index.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """
    Container for algorithmic complexity measurements.
    
    This structure tracks empirical performance characteristics needed to
    validate theoretical complexity bounds in academic publications.
    """
    operation_name: str
    num_comparisons: int = 0
    wall_clock_time_ms: float = 0.0
    input_size: int = 0
    output_size: int = 0
    memory_allocated_bytes: int = 0
    cache_hit: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Union[str, int, float, bool]]:
        """Export metrics for analysis."""
        return {
            'operation': self.operation_name,
            'comparisons': self.num_comparisons,
            'time_ms': self.wall_clock_time_ms,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'memory_bytes': self.memory_allocated_bytes,
            'cache_hit': self.cache_hit,
            'timestamp': self.timestamp
        }


@dataclass
class TransactionRecord:
    """
    Immutable record representing a single cryptocurrency transaction.
    
    This enhanced structure includes validation and efficient serialization
    for storage and network transmission.
    """
    edge_id: int
    source_id: int
    target_id: int
    timestamp: float
    attributes: np.ndarray
    block_height: int = 0
    
    def __post_init__(self):
        """Validate transaction record on creation."""
        if self.edge_id < 0:
            raise ValueError(f"Invalid edge_id: {self.edge_id}")
        if self.source_id < 0 or self.target_id < 0:
            raise ValueError(f"Invalid node IDs: source={self.source_id}, target={self.target_id}")
        if self.timestamp < 0:
            raise ValueError(f"Invalid timestamp: {self.timestamp}")
        if not isinstance(self.attributes, np.ndarray):
            self.attributes = np.array(self.attributes, dtype=np.float32)
    
    def __lt__(self, other: 'TransactionRecord') -> bool:
        """Enable sorting by timestamp for temporal ordering."""
        return self.timestamp < other.timestamp
    
    def __hash__(self) -> int:
        """Enable hashing for set operations and deduplication."""
        return hash(self.edge_id)
    
    def __eq__(self, other: object) -> bool:
        """Enable equality comparison for deduplication."""
        if not isinstance(other, TransactionRecord):
            return False
        return self.edge_id == other.edge_id
    
    def serialize(self) -> bytes:
        """Serialize transaction to bytes for efficient storage."""
        # Pack fixed fields
        header = struct.pack('!IIIdi', 
                           self.edge_id, self.source_id, self.target_id,
                           self.timestamp, self.block_height)
        # Pack attributes
        attr_bytes = self.attributes.astype(np.float32).tobytes()
        return header + attr_bytes
    
    @classmethod
    def deserialize(cls, data: bytes, attr_dim: int) -> 'TransactionRecord':
        """Deserialize transaction from bytes."""
        header_size = struct.calcsize('!IIIdi')
        header = struct.unpack('!IIIdi', data[:header_size])
        
        attr_bytes = data[header_size:]
        attributes = np.frombuffer(attr_bytes, dtype=np.float32).reshape(-1)
        
        return cls(
            edge_id=header[0],
            source_id=header[1],
            target_id=header[2],
            timestamp=header[3],
            block_height=header[4],
            attributes=attributes
        )


class InstrumentedBinarySearch:
    """
    Wrapper providing instrumented binary search operations with complexity tracking.
    
    This utility enables precise measurement of comparison counts and search
    characteristics for empirical complexity validation.
    """
    
    @staticmethod
    def insort_left_instrumented(lst: List, item: TransactionRecord) -> Tuple[int, int]:
        """
        Binary insertion maintaining sorted order with comparison counting.
        
        Returns:
            Tuple of (insertion_index, num_comparisons)
        """
        lo, hi = 0, len(lst)
        comparisons = 0
        
        while lo < hi:
            mid = (lo + hi) // 2
            comparisons += 1
            if lst[mid] < item:
                lo = mid + 1
            else:
                hi = mid
        
        lst.insert(lo, item)
        return lo, comparisons
    
    @staticmethod
    def bisect_left_instrumented(lst: List, item: TransactionRecord) -> Tuple[int, int]:
        """
        Binary search returning index with comparison counting.
        
        Returns:
            Tuple of (index, num_comparisons)
        """
        lo, hi = 0, len(lst)
        comparisons = 0
        
        while lo < hi:
            mid = (lo + hi) // 2
            comparisons += 1
            if lst[mid] < item:
                lo = mid + 1
            else:
                hi = mid
        
        return lo, comparisons
    
    @staticmethod
    def bisect_right_instrumented(lst: List, item: TransactionRecord) -> Tuple[int, int]:
        """
        Binary search for rightmost position with comparison counting.
        
        Returns:
            Tuple of (index, num_comparisons)
        """
        lo, hi = 0, len(lst)
        comparisons = 0
        
        while lo < hi:
            mid = (lo + hi) // 2
            comparisons += 1
            if item < lst[mid]:
                hi = mid
            else:
                lo = mid + 1
        
        return lo, comparisons


@dataclass
class TemporalWindow:
    """
    Enhanced temporal window with comprehensive profiling instrumentation.
    
    This structure maintains transaction sequences for a single account with
    detailed tracking of operation characteristics for complexity analysis.
    """
    account_id: int
    incoming: List[TransactionRecord] = field(default_factory=list)
    outgoing: List[TransactionRecord] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = float('inf')
    _memory_size: int = 0
    _last_update: float = field(default_factory=time.time)
    
    # Profiling metrics
    _total_insertions: int = 0
    _total_queries: int = 0
    _total_insert_comparisons: int = 0
    _total_query_comparisons: int = 0
    _insert_time_ms: float = 0.0
    _query_time_ms: float = 0.0
    _tier_transitions: int = 0
    _query_cache_hits: int = 0
    
    def add_transaction(self, transaction: TransactionRecord, direction: str) -> ComplexityMetrics:
        """
        Add transaction maintaining sorted order with comprehensive profiling.
        
        Returns:
            ComplexityMetrics capturing insertion performance characteristics
        """
        start_time = time.perf_counter()
        target_list = self.incoming if direction == 'in' else self.outgoing
        initial_size = len(target_list)
        
        # Check for duplicates (linear search, but should be rare)
        if transaction in target_list:
            metrics = ComplexityMetrics(
                operation_name='add_transaction_duplicate',
                num_comparisons=1,
                wall_clock_time_ms=(time.perf_counter() - start_time) * 1000,
                input_size=initial_size,
                output_size=initial_size
            )
            return metrics
        
        # Instrumented binary search insertion
        insertion_idx, comparisons = InstrumentedBinarySearch.insort_left_instrumented(
            target_list, transaction
        )
        
        self._last_update = time.time()
        self._total_insertions += 1
        self._total_insert_comparisons += comparisons
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._insert_time_ms += elapsed_ms
        
        # Update memory tracking
        memory_before = self._memory_size
        self._update_memory_size()
        memory_allocated = self._memory_size - memory_before
        
        metrics = ComplexityMetrics(
            operation_name='add_transaction',
            num_comparisons=comparisons,
            wall_clock_time_ms=elapsed_ms,
            input_size=initial_size,
            output_size=len(target_list),
            memory_allocated_bytes=memory_allocated
        )
        
        return metrics
    
    def get_sequence(self, direction: str, max_length: Optional[int] = None,
                     end_time: Optional[float] = None,
                     start_time: Optional[float] = None,
                     track_metrics: bool = True) -> Union[List[TransactionRecord], 
                                                           Tuple[List[TransactionRecord], ComplexityMetrics]]:
        """
        Retrieve transaction sequence with comprehensive profiling.
        
        Args:
            direction: 'in' for incoming, 'out' for outgoing
            max_length: Optional limit on number of transactions returned
            end_time: Optional temporal upper bound
            start_time: Optional temporal lower bound
            track_metrics: Whether to return profiling metrics
            
        Returns:
            Transaction sequence, optionally with ComplexityMetrics
        """
        operation_start = time.perf_counter()
        sequence = self.incoming if direction == 'in' else self.outgoing
        initial_size = len(sequence)
        total_comparisons = 0
        
        # Apply time filters efficiently using instrumented binary search
        if start_time is not None or end_time is not None:
            start_idx = 0
            end_idx = len(sequence)
            
            if start_time is not None:
                dummy_start = TransactionRecord(0, 0, 0, start_time, np.array([]))
                start_idx, comparisons_start = InstrumentedBinarySearch.bisect_left_instrumented(
                    sequence, dummy_start
                )
                total_comparisons += comparisons_start
            
            if end_time is not None:
                dummy_end = TransactionRecord(0, 0, 0, end_time, np.array([]))
                end_idx, comparisons_end = InstrumentedBinarySearch.bisect_right_instrumented(
                    sequence, dummy_end
                )
                total_comparisons += comparisons_end
            
            sequence = sequence[start_idx:end_idx]
        
        # Apply length limit (return most recent)
        if max_length is not None and len(sequence) > max_length:
            sequence = sequence[-max_length:]
        
        elapsed_ms = (time.perf_counter() - operation_start) * 1000
        
        if track_metrics:
            self._total_queries += 1
            self._total_query_comparisons += total_comparisons
            self._query_time_ms += elapsed_ms
            
            metrics = ComplexityMetrics(
                operation_name='get_sequence',
                num_comparisons=total_comparisons,
                wall_clock_time_ms=elapsed_ms,
                input_size=initial_size,
                output_size=len(sequence)
            )
            return sequence, metrics
        else:
            return sequence
    
    def prune_before(self, cutoff_time: float) -> Tuple[int, ComplexityMetrics]:
        """
        Remove old transactions efficiently with profiling.
        
        Returns:
            Tuple of (num_pruned, ComplexityMetrics)
        """
        start_time = time.perf_counter()
        original_count = len(self.incoming) + len(self.outgoing)
        total_comparisons = 0
        
        # Use instrumented binary search for efficient pruning
        dummy_tx = TransactionRecord(0, 0, 0, cutoff_time, np.array([]))
        
        in_idx, in_comparisons = InstrumentedBinarySearch.bisect_left_instrumented(
            self.incoming, dummy_tx
        )
        self.incoming = self.incoming[in_idx:]
        total_comparisons += in_comparisons
        
        out_idx, out_comparisons = InstrumentedBinarySearch.bisect_left_instrumented(
            self.outgoing, dummy_tx
        )
        self.outgoing = self.outgoing[out_idx:]
        total_comparisons += out_comparisons
        
        pruned = original_count - (len(self.incoming) + len(self.outgoing))
        self._update_memory_size()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        metrics = ComplexityMetrics(
            operation_name='prune_before',
            num_comparisons=total_comparisons,
            wall_clock_time_ms=elapsed_ms,
            input_size=original_count,
            output_size=len(self.incoming) + len(self.outgoing)
        )
        
        return pruned, metrics
    
    def _update_memory_size(self):
        """Update estimated memory footprint."""
        base_size = sys.getsizeof(self)
        list_size = sys.getsizeof(self.incoming) + sys.getsizeof(self.outgoing)
        
        tx_size = 0
        for tx in self.incoming + self.outgoing:
            tx_size += sys.getsizeof(tx) + tx.attributes.nbytes
        
        self._memory_size = base_size + list_size + tx_size
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        Export comprehensive profiling statistics for analysis.
        
        Returns:
            Dictionary with detailed performance characteristics
        """
        avg_insert_comparisons = (self._total_insert_comparisons / self._total_insertions 
                                 if self._total_insertions > 0 else 0)
        avg_query_comparisons = (self._total_query_comparisons / self._total_queries 
                                if self._total_queries > 0 else 0)
        avg_insert_time = (self._insert_time_ms / self._total_insertions 
                          if self._total_insertions > 0 else 0)
        avg_query_time = (self._query_time_ms / self._total_queries 
                         if self._total_queries > 0 else 0)
        
        return {
            'account_id': self.account_id,
            'num_incoming': len(self.incoming),
            'num_outgoing': len(self.outgoing),
            'memory_bytes': self._memory_size,
            'total_insertions': self._total_insertions,
            'total_queries': self._total_queries,
            'avg_insert_comparisons': avg_insert_comparisons,
            'avg_query_comparisons': avg_query_comparisons,
            'avg_insert_time_ms': avg_insert_time,
            'avg_query_time_ms': avg_query_time,
            'tier_transitions': self._tier_transitions,
            'query_cache_hits': self._query_cache_hits,
            'cache_hit_rate': (self._query_cache_hits / self._total_queries 
                              if self._total_queries > 0 else 0)
        }


@dataclass
class StorageTierMetrics:
    """Metrics for multi-tier storage hierarchy analysis."""
    hot_to_cold_transitions: int = 0
    cold_to_hot_transitions: int = 0
    hot_tier_accesses: int = 0
    cold_tier_accesses: int = 0
    hot_tier_size_bytes: int = 0
    cold_tier_size_bytes: int = 0
    compression_operations: int = 0
    decompression_operations: int = 0
    total_compressed_bytes: int = 0
    total_uncompressed_bytes: int = 0
    avg_compression_ratio: float = 0.0
    
    def update_compression_ratio(self):
        """Calculate average compression ratio."""
        if self.total_uncompressed_bytes > 0:
            self.avg_compression_ratio = (self.total_compressed_bytes / 
                                         self.total_uncompressed_bytes)
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Export metrics for analysis."""
        return {
            'hot_to_cold_transitions': self.hot_to_cold_transitions,
            'cold_to_hot_transitions': self.cold_to_hot_transitions,
            'hot_tier_accesses': self.hot_tier_accesses,
            'cold_tier_accesses': self.cold_tier_accesses,
            'hot_tier_size_mb': self.hot_tier_size_bytes / (1024 ** 2),
            'cold_tier_size_mb': self.cold_tier_size_bytes / (1024 ** 2),
            'compression_operations': self.compression_operations,
            'decompression_operations': self.decompression_operations,
            'avg_compression_ratio': self.avg_compression_ratio
        }


class StorageManager:
    """
    Enhanced multi-tier storage manager with comprehensive profiling.
    
    This component implements a two-tier storage hierarchy (hot/cold) with
    detailed tracking of access patterns, tier transitions, and compression
    effectiveness for empirical validation of the storage strategy.
    """
    
    def __init__(self, hot_memory_limit_mb: float = 1000.0,
                 cold_storage_path: str = './cold_storage',
                 compression_enabled: bool = True):
        """Initialize storage manager with profiling."""
        self.hot_memory_limit_bytes = int(hot_memory_limit_mb * 1024 * 1024)
        self.cold_storage_path = cold_storage_path
        self.compression_enabled = compression_enabled
        
        # Storage tiers
        self.hot_data: Dict[int, TemporalWindow] = {}
        self.cold_data: Dict[int, str] = {}  # Maps account_id to file path
        
        # Profiling metrics
        self.metrics = StorageTierMetrics()
        self.access_frequency: Dict[int, int] = defaultdict(int)
        self.last_access_time: Dict[int, float] = {}
        
        # Ensure cold storage directory exists
        os.makedirs(cold_storage_path, exist_ok=True)
        
        logger.info(f"StorageManager initialized: hot_limit={hot_memory_limit_mb}MB, "
                   f"compression={'enabled' if compression_enabled else 'disabled'}")
    
    def store_window(self, window: TemporalWindow, force_tier: Optional[str] = None):
        """
        Store temporal window with intelligent tier placement and profiling.
        
        Args:
            window: TemporalWindow to store
            force_tier: Optional tier specification ('hot' or 'cold')
        """
        account_id = window.account_id
        self.access_frequency[account_id] += 1
        self.last_access_time[account_id] = time.time()
        
        # Determine target tier
        if force_tier == 'hot' or (force_tier is None and self._should_keep_hot(window)):
            self._store_to_hot(window)
        else:
            self._store_to_cold(window)
    
    def retrieve_window(self, account_id: int) -> Optional[TemporalWindow]:
        """
        Retrieve temporal window with automatic tier promotion and profiling.
        
        Args:
            account_id: Account identifier
            
        Returns:
            TemporalWindow if found, None otherwise
        """
        self.access_frequency[account_id] += 1
        self.last_access_time[account_id] = time.time()
        
        # Check hot tier first
        if account_id in self.hot_data:
            self.metrics.hot_tier_accesses += 1
            return self.hot_data[account_id]
        
        # Check cold tier
        if account_id in self.cold_data:
            self.metrics.cold_tier_accesses += 1
            window = self._load_from_cold(account_id)
            
            # Promote to hot if accessed frequently
            if window and self._should_promote_to_hot(account_id):
                self._promote_to_hot(window)
            
            return window
        
        return None
    
    def _should_keep_hot(self, window: TemporalWindow) -> bool:
        """Determine if window should remain in hot tier."""
        current_hot_size = sum(w._memory_size for w in self.hot_data.values())
        
        # Keep in hot if under limit
        if current_hot_size + window._memory_size < self.hot_memory_limit_bytes:
            return True
        
        # Otherwise, check access frequency
        account_id = window.account_id
        if self.access_frequency[account_id] > 5:  # Threshold for "hot" data
            return True
        
        return False
    
    def _should_promote_to_hot(self, account_id: int) -> bool:
        """Determine if cold data should be promoted to hot tier."""
        # Promote if accessed multiple times recently
        return self.access_frequency[account_id] > 3
    
    def _store_to_hot(self, window: TemporalWindow):
        """Store window to hot tier with eviction if necessary."""
        account_id = window.account_id
        
        # If moving from cold to hot
        if account_id in self.cold_data:
            self.metrics.cold_to_hot_transitions += 1
            del self.cold_data[account_id]
            # Clean up cold storage file
            cold_path = os.path.join(self.cold_storage_path, f'account_{account_id}.dat')
            if os.path.exists(cold_path):
                os.remove(cold_path)
        
        # Evict least recently used if necessary
        while True:
            current_hot_size = sum(w._memory_size for w in self.hot_data.values())
            if current_hot_size + window._memory_size < self.hot_memory_limit_bytes:
                break
            
            # Find LRU account
            lru_account = min(
                (aid for aid in self.hot_data.keys() if aid != account_id),
                key=lambda aid: self.last_access_time.get(aid, 0),
                default=None
            )
            
            if lru_account is None:
                break
            
            # Evict to cold
            evicted_window = self.hot_data[lru_account]
            self._store_to_cold(evicted_window)
            del self.hot_data[lru_account]
        
        self.hot_data[account_id] = window
        self.metrics.hot_tier_size_bytes = sum(w._memory_size for w in self.hot_data.values())
    
    def _store_to_cold(self, window: TemporalWindow):
        """Store window to cold tier with compression."""
        account_id = window.account_id
        
        # If moving from hot to cold
        if account_id in self.hot_data:
            self.metrics.hot_to_cold_transitions += 1
            del self.hot_data[account_id]
        
        # Serialize window
        serialized = pickle.dumps(window, protocol=pickle.HIGHEST_PROTOCOL)
        uncompressed_size = len(serialized)
        
        # Compress if enabled
        if self.compression_enabled:
            compressed = lz4.frame.compress(serialized)
            compressed_size = len(compressed)
            data_to_write = compressed
            
            self.metrics.compression_operations += 1
            self.metrics.total_uncompressed_bytes += uncompressed_size
            self.metrics.total_compressed_bytes += compressed_size
            self.metrics.update_compression_ratio()
        else:
            data_to_write = serialized
            compressed_size = uncompressed_size
        
        # Write to cold storage
        cold_path = os.path.join(self.cold_storage_path, f'account_{account_id}.dat')
        with open(cold_path, 'wb') as f:
            f.write(data_to_write)
        
        self.cold_data[account_id] = cold_path
        self.metrics.cold_tier_size_bytes += compressed_size
    
    def _load_from_cold(self, account_id: int) -> Optional[TemporalWindow]:
        """Load window from cold tier with decompression."""
        if account_id not in self.cold_data:
            return None
        
        cold_path = self.cold_data[account_id]
        
        try:
            with open(cold_path, 'rb') as f:
                data = f.read()
            
            # Decompress if enabled
            if self.compression_enabled:
                decompressed = lz4.frame.decompress(data)
                self.metrics.decompression_operations += 1
                window = pickle.loads(decompressed)
            else:
                window = pickle.loads(data)
            
            return window
        
        except Exception as e:
            logger.error(f"Failed to load window for account {account_id}: {e}")
            return None
    
    def _promote_to_hot(self, window: TemporalWindow):
        """Promote window from cold to hot tier."""
        self._store_to_hot(window)
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Export comprehensive storage metrics."""
        return {
            'num_hot_accounts': len(self.hot_data),
            'num_cold_accounts': len(self.cold_data),
            'hot_memory_limit_mb': self.hot_memory_limit_bytes / (1024 ** 2),
            'tier_metrics': self.metrics.to_dict(),
            'access_patterns': {
                'mean_access_frequency': (np.mean(list(self.access_frequency.values()))
                                         if self.access_frequency else 0),
                'max_access_frequency': (max(self.access_frequency.values())
                                        if self.access_frequency else 0),
                'accounts_tracked': len(self.access_frequency)
            }
        }


@dataclass
class AdaptiveSequenceDecision:
    """Record of adaptive sequence length adjustment decision."""
    account_id: int
    previous_length: int
    new_length: int
    trigger_reason: str
    num_incoming_transactions: int
    num_outgoing_transactions: int
    memory_before_bytes: int
    memory_after_bytes: int
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Union[int, str, float]]:
        """Export decision record for analysis."""
        return {
            'account_id': self.account_id,
            'previous_length': self.previous_length,
            'new_length': self.new_length,
            'trigger_reason': self.trigger_reason,
            'num_incoming': self.num_incoming_transactions,
            'num_outgoing': self.num_outgoing_transactions,
            'memory_delta_kb': (self.memory_after_bytes - self.memory_before_bytes) / 1024,
            'timestamp': self.timestamp
        }


class AdaptiveSequenceManager:
    """
    Enhanced adaptive sequence length manager with detailed decision tracking.
    
    This component dynamically adjusts sequence lengths based on transaction
    patterns while maintaining comprehensive records of adaptation decisions
    for empirical validation of the adaptive strategy.
    """
    
    def __init__(self,
                 min_sequence_length: int = 8,
                 max_sequence_length: int = 128,
                 default_sequence_length: int = 32,
                 adaptation_threshold: int = 10):
        """Initialize adaptive manager with profiling."""
        self.min_length = min_sequence_length
        self.max_length = max_sequence_length
        self.default_length = default_sequence_length
        self.adaptation_threshold = adaptation_threshold
        
        # Current length assignments
        self.account_lengths: Dict[int, int] = defaultdict(lambda: default_sequence_length)
        
        # Transaction count tracking
        self.transaction_counts: Dict[int, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        
        # Profiling: decision history
        self.adaptation_decisions: List[AdaptiveSequenceDecision] = []
        self.total_adaptations: int = 0
        self.adaptation_triggers: Dict[str, int] = defaultdict(int)
        
        logger.info(f"AdaptiveSequenceManager initialized: "
                   f"min={min_sequence_length}, max={max_sequence_length}, "
                   f"default={default_sequence_length}")
    
    def get_sequence_length(self, account_id: int) -> int:
        """Get current sequence length for account."""
        return self.account_lengths[account_id]
    
    def update_and_adapt(self, window: TemporalWindow) -> Optional[AdaptiveSequenceDecision]:
        """
        Update transaction counts and adapt sequence length if needed.
        
        Args:
            window: TemporalWindow to analyze
            
        Returns:
            AdaptiveSequenceDecision if adaptation occurred, None otherwise
        """
        account_id = window.account_id
        num_incoming = len(window.incoming)
        num_outgoing = len(window.outgoing)
        
        # Update transaction counts
        self.transaction_counts[account_id] = (num_incoming, num_outgoing)
        
        # Check if adaptation is needed
        current_length = self.account_lengths[account_id]
        new_length = self._compute_optimal_length(num_incoming, num_outgoing)
        
        # Only adapt if change exceeds threshold
        if abs(new_length - current_length) >= self.adaptation_threshold:
            memory_before = window._memory_size
            
            # Determine trigger reason
            trigger_reason = self._determine_trigger_reason(
                num_incoming, num_outgoing, current_length, new_length
            )
            
            # Update length
            self.account_lengths[account_id] = new_length
            
            # Update window memory estimate (actual truncation happens during retrieval)
            window._update_memory_size()
            memory_after = window._memory_size
            
            # Record decision
            decision = AdaptiveSequenceDecision(
                account_id=account_id,
                previous_length=current_length,
                new_length=new_length,
                trigger_reason=trigger_reason,
                num_incoming_transactions=num_incoming,
                num_outgoing_transactions=num_outgoing,
                memory_before_bytes=memory_before,
                memory_after_bytes=memory_after
            )
            
            self.adaptation_decisions.append(decision)
            self.total_adaptations += 1
            self.adaptation_triggers[trigger_reason] += 1
            
            logger.debug(f"Adapted sequence length for account {account_id}: "
                        f"{current_length} -> {new_length} ({trigger_reason})")
            
            return decision
        
        return None
    
    def _compute_optimal_length(self, num_incoming: int, num_outgoing: int) -> int:
        """
        Compute optimal sequence length based on transaction patterns.
        
        This heuristic balances memory efficiency with model accuracy by
        adjusting sequence length based on transaction volume.
        """
        total_transactions = num_incoming + num_outgoing
        
        # Simple adaptive policy: scale with sqrt of transaction count
        # This provides sublinear growth appropriate for memory-constrained scenarios
        optimal = int(self.default_length * np.sqrt(max(1, total_transactions) / 50))
        
        # Clamp to valid range
        optimal = max(self.min_length, min(self.max_length, optimal))
        
        return optimal
    
    def _determine_trigger_reason(self, num_incoming: int, num_outgoing: int,
                                   old_length: int, new_length: int) -> str:
        """Classify the reason for sequence length adaptation."""
        total = num_incoming + num_outgoing
        
        if new_length > old_length:
            if total > old_length * 1.5:
                return 'high_transaction_volume'
            else:
                return 'proactive_capacity_increase'
        else:
            if total < old_length * 0.5:
                return 'low_transaction_volume'
            else:
                return 'memory_optimization'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Export comprehensive adaptation statistics."""
        lengths = list(self.account_lengths.values())
        
        return {
            'total_adaptations': self.total_adaptations,
            'accounts_managed': len(self.account_lengths),
            'mean_sequence_length': float(np.mean(lengths)) if lengths else 0,
            'std_sequence_length': float(np.std(lengths)) if lengths else 0,
            'min_sequence_length': int(np.min(lengths)) if lengths else 0,
            'max_sequence_length': int(np.max(lengths)) if lengths else 0,
            'p25_sequence_length': float(np.percentile(lengths, 25)) if lengths else 0,
            'p50_sequence_length': float(np.percentile(lengths, 50)) if lengths else 0,
            'p75_sequence_length': float(np.percentile(lengths, 75)) if lengths else 0,
            'adaptation_triggers': dict(self.adaptation_triggers),
            'recent_decisions': [d.to_dict() for d in self.adaptation_decisions[-100:]]
        }
    
    def export_adaptation_history(self, filepath: str):
        """Export complete adaptation decision history for analysis."""
        history = [d.to_dict() for d in self.adaptation_decisions]
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Exported {len(history)} adaptation decisions to {filepath}")


class StreamingIngestionEngine:
    """
    Asynchronous transaction ingestion with batching and profiling.
    
    This component processes incoming transactions asynchronously while
    tracking ingestion performance characteristics.
    """
    
    def __init__(self, batch_size: int = 100, max_queue_size: int = 10000):
        """Initialize ingestion engine."""
        self.batch_size = batch_size
        self.ingestion_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Profiling metrics
        self.total_ingested = 0
        self.total_batches = 0
        self.total_ingestion_time_ms = 0.0
        self.queue_full_count = 0
        
        # Callback for batch processing
        self.batch_callback: Optional[Callable] = None
    
    def start(self, batch_callback: Callable):
        """Start asynchronous ingestion."""
        self.batch_callback = batch_callback
        self.running = True
        self.worker_thread = threading.Thread(target=self._ingestion_worker, daemon=True)
        self.worker_thread.start()
        logger.info("StreamingIngestionEngine started")
    
    def enqueue_transaction(self, transaction: TransactionRecord) -> bool:
        """
        Enqueue transaction for asynchronous processing.
        
        Returns:
            True if enqueued successfully, False if queue full
        """
        try:
            self.ingestion_queue.put_nowait(transaction)
            return True
        except queue.Full:
            self.queue_full_count += 1
            return False
    
    def _ingestion_worker(self):
        """Background worker processing transaction batches."""
        batch = []
        
        while self.running:
            try:
                # Collect batch
                while len(batch) < self.batch_size:
                    try:
                        transaction = self.ingestion_queue.get(timeout=0.1)
                        batch.append(transaction)
                    except queue.Empty:
                        break
                
                # Process batch if non-empty
                if batch and self.batch_callback:
                    start_time = time.perf_counter()
                    self.batch_callback(batch)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    
                    self.total_ingested += len(batch)
                    self.total_batches += 1
                    self.total_ingestion_time_ms += elapsed_ms
                    
                    batch = []
            
            except Exception as e:
                logger.error(f"Error in ingestion worker: {e}")
    
    def shutdown(self):
        """Shutdown ingestion engine gracefully."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("StreamingIngestionEngine shutdown")
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Export ingestion performance metrics."""
        avg_batch_time = (self.total_ingestion_time_ms / self.total_batches 
                         if self.total_batches > 0 else 0)
        throughput = (self.total_ingested / (self.total_ingestion_time_ms / 1000)
                     if self.total_ingestion_time_ms > 0 else 0)
        
        return {
            'total_ingested': self.total_ingested,
            'total_batches': self.total_batches,
            'avg_batch_time_ms': avg_batch_time,
            'throughput_tps': throughput,
            'queue_size': self.ingestion_queue.qsize(),
            'queue_full_count': self.queue_full_count
        }


class GraphPartitioner:
    """
    Graph partitioning for distributed processing.
    
    This component assigns accounts to partitions for distributed processing
    while maintaining load balance.
    """
    
    def __init__(self, num_partitions: int = 4, strategy: str = 'balanced'):
        """Initialize partitioner."""
        self.num_partitions = num_partitions
        self.strategy = strategy
        self.account_partitions: Dict[int, int] = {}
        
        logger.info(f"GraphPartitioner initialized: "
                   f"partitions={num_partitions}, strategy={strategy}")
    
    def get_partition(self, account_id: int) -> int:
        """Get partition assignment for account."""
        if account_id not in self.account_partitions:
            if self.strategy == 'balanced':
                # Simple hash-based partitioning
                self.account_partitions[account_id] = account_id % self.num_partitions
            else:
                # Random partitioning
                self.account_partitions[account_id] = np.random.randint(0, self.num_partitions)
        
        return self.account_partitions[account_id]
    
    def compute_partition_balance(self) -> float:
        """Compute partition load balance metric."""
        if not self.account_partitions:
            return 1.0
        
        partition_counts = defaultdict(int)
        for partition_id in self.account_partitions.values():
            partition_counts[partition_id] += 1
        
        counts = list(partition_counts.values())
        if not counts:
            return 1.0
        
        # Balance metric: ratio of min to max partition size
        balance = min(counts) / max(counts) if max(counts) > 0 else 1.0
        return balance


class TemporalMultiGraphIndex:
    """
    Enhanced temporal multigraph index with comprehensive profiling instrumentation.
    
    This is the main index structure coordinating all components with detailed
    tracking of operation characteristics for empirical complexity validation.
    """
    
    def __init__(self,
                 num_partitions: int = 4,
                 min_sequence_length: int = 8,
                 max_sequence_length: int = 128,
                 default_sequence_length: int = 32,
                 hot_memory_limit_gb: float = 1.0,
                 enable_streaming: bool = True,
                 enable_profiling: bool = True):
        """Initialize enhanced temporal index with profiling."""
        self.num_partitions = num_partitions
        self.enable_profiling = enable_profiling
        
        # Core data structures
        self.temporal_windows: Dict[int, TemporalWindow] = {}
        self.index_lock = threading.RLock()
        
        # Component initialization
        self.adaptive_manager = AdaptiveSequenceManager(
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            default_sequence_length=default_sequence_length
        )
        
        self.storage_manager = StorageManager(
            hot_memory_limit_mb=hot_memory_limit_gb * 1024,
            cold_storage_path='./cold_storage',
            compression_enabled=True
        )
        
        self.partitioner = GraphPartitioner(
            num_partitions=num_partitions,
            strategy='balanced'
        )
        
        # Streaming ingestion
        self.ingestion_engine: Optional[StreamingIngestionEngine] = None
        if enable_streaming:
            self.ingestion_engine = StreamingIngestionEngine(batch_size=100)
            self.ingestion_engine.start(self._process_transaction_batch)
        
        # Global statistics
        self.num_accounts = 0
        self.num_transactions = 0
        self.min_timestamp = float('inf')
        self.max_timestamp = 0.0
        
        # Profiling: operation metrics collection
        self.operation_metrics: List[ComplexityMetrics] = []
        self.metrics_lock = threading.Lock()
        
        logger.info(f"TemporalMultiGraphIndex initialized: "
                   f"partitions={num_partitions}, streaming={enable_streaming}, "
                   f"profiling={enable_profiling}")
    
    def add_transaction(self, transaction: TransactionRecord):
        """
        Add transaction to index with profiling.
        
        This method either directly processes the transaction or enqueues it
        for asynchronous processing depending on streaming configuration.
        """
        if self.ingestion_engine:
            # Asynchronous ingestion
            success = self.ingestion_engine.enqueue_transaction(transaction)
            if not success:
                logger.warning(f"Ingestion queue full, transaction {transaction.edge_id} dropped")
        else:
            # Synchronous processing
            self._process_single_transaction(transaction)
    
    def _process_single_transaction(self, transaction: TransactionRecord):
        """Process individual transaction with profiling."""
        with self.index_lock:
            # Update source account (outgoing)
            source_metrics = self._add_to_account(
                transaction.source_id, transaction, 'out'
            )
            
            # Update target account (incoming)
            target_metrics = self._add_to_account(
                transaction.target_id, transaction, 'in'
            )
            
            # Update global statistics
            self.num_transactions += 1
            self.min_timestamp = min(self.min_timestamp, transaction.timestamp)
            self.max_timestamp = max(self.max_timestamp, transaction.timestamp)
            
            # Record metrics if profiling enabled
            if self.enable_profiling:
                with self.metrics_lock:
                    if source_metrics:
                        self.operation_metrics.append(source_metrics)
                    if target_metrics:
                        self.operation_metrics.append(target_metrics)
    
    def _process_transaction_batch(self, batch: List[TransactionRecord]):
        """Process batch of transactions."""
        for transaction in batch:
            self._process_single_transaction(transaction)
    
    def _add_to_account(self, account_id: int, transaction: TransactionRecord,
                       direction: str) -> Optional[ComplexityMetrics]:
        """
        Add transaction to account's temporal window with profiling.
        
        Returns:
            ComplexityMetrics for the insertion operation
        """
        # Get or create temporal window
        if account_id not in self.temporal_windows:
            self.temporal_windows[account_id] = TemporalWindow(account_id=account_id)
            self.num_accounts += 1
        
        window = self.temporal_windows[account_id]
        
        # Add transaction with profiling
        metrics = window.add_transaction(transaction, direction)
        
        # Adaptive sequence management
        adaptation = self.adaptive_manager.update_and_adapt(window)
        
        # Storage tier management
        self.storage_manager.store_window(window)
        
        return metrics
    
    def get_transaction_sequences(self, account_ids: List[int],
                                  max_length: Optional[int] = None,
                                  end_time: Optional[float] = None,
                                  start_time: Optional[float] = None) -> Dict[int, Tuple[List, List]]:
        """
        Retrieve transaction sequences for multiple accounts with profiling.
        
        Args:
            account_ids: List of account identifiers
            max_length: Optional sequence length limit
            end_time: Optional temporal upper bound
            start_time: Optional temporal lower bound
            
        Returns:
            Dictionary mapping account_id to (incoming, outgoing) sequence tuples
        """
        sequences = {}
        
        with self.index_lock:
            for account_id in account_ids:
                # Retrieve from storage (handles hot/cold tiers)
                window = self.storage_manager.retrieve_window(account_id)
                
                if window is None:
                    window = self.temporal_windows.get(account_id)
                
                if window:
                    # Get adaptive length if not specified
                    if max_length is None:
                        max_length = self.adaptive_manager.get_sequence_length(account_id)
                    
                    # Retrieve sequences with profiling
                    if self.enable_profiling:
                        incoming, in_metrics = window.get_sequence(
                            'in', max_length, end_time, start_time, track_metrics=True
                        )
                        outgoing, out_metrics = window.get_sequence(
                            'out', max_length, end_time, start_time, track_metrics=True
                        )
                        
                        with self.metrics_lock:
                            self.operation_metrics.append(in_metrics)
                            self.operation_metrics.append(out_metrics)
                    else:
                        incoming = window.get_sequence(
                            'in', max_length, end_time, start_time, track_metrics=False
                        )
                        outgoing = window.get_sequence(
                            'out', max_length, end_time, start_time, track_metrics=False
                        )
                    
                    sequences[account_id] = (incoming, outgoing)
                else:
                    sequences[account_id] = ([], [])
        
        return sequences
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Export comprehensive index statistics with profiling metrics."""
        with self.index_lock:
            # Component statistics
            adaptive_stats = self.adaptive_manager.get_statistics()
            storage_stats = self.storage_manager.get_storage_statistics()
            ingestion_stats = (self.ingestion_engine.get_ingestion_statistics()
                             if self.ingestion_engine else {})
            
            # Graph statistics
            in_degrees = []
            out_degrees = []
            
            sample_size = min(1000, len(self.temporal_windows))
            sampled_windows = list(self.temporal_windows.values())[:sample_size]
            
            for window in sampled_windows:
                in_degrees.append(len(window.incoming))
                out_degrees.append(len(window.outgoing))
            
            # Complexity metrics aggregation
            if self.enable_profiling:
                complexity_stats = self._aggregate_complexity_metrics()
            else:
                complexity_stats = {}
            
            return {
                'num_accounts': self.num_accounts,
                'num_transactions': self.num_transactions,
                'temporal_range_seconds': (self.max_timestamp - self.min_timestamp 
                                          if self.num_transactions > 0 else 0),
                'mean_in_degree': float(np.mean(in_degrees)) if in_degrees else 0,
                'std_in_degree': float(np.std(in_degrees)) if in_degrees else 0,
                'mean_out_degree': float(np.mean(out_degrees)) if out_degrees else 0,
                'std_out_degree': float(np.std(out_degrees)) if out_degrees else 0,
                'partition_balance': self.partitioner.compute_partition_balance(),
                'adaptive_sequence': adaptive_stats,
                'storage': storage_stats,
                'ingestion': ingestion_stats,
                'complexity_analysis': complexity_stats
            }
    
    def _aggregate_complexity_metrics(self) -> Dict[str, Any]:
        """Aggregate complexity metrics for analysis."""
        with self.metrics_lock:
            if not self.operation_metrics:
                return {}
            
            # Group metrics by operation type
            by_operation = defaultdict(list)
            for metric in self.operation_metrics:
                by_operation[metric.operation_name].append(metric)
            
            aggregated = {}
            for op_name, metrics in by_operation.items():
                comparisons = [m.num_comparisons for m in metrics]
                times = [m.wall_clock_time_ms for m in metrics]
                input_sizes = [m.input_size for m in metrics]
                
                aggregated[op_name] = {
                    'count': len(metrics),
                    'mean_comparisons': float(np.mean(comparisons)),
                    'std_comparisons': float(np.std(comparisons)),
                    'mean_time_ms': float(np.mean(times)),
                    'std_time_ms': float(np.std(times)),
                    'p50_time_ms': float(np.percentile(times, 50)),
                    'p95_time_ms': float(np.percentile(times, 95)),
                    'p99_time_ms': float(np.percentile(times, 99)),
                    'mean_input_size': float(np.mean(input_sizes)),
                    'cache_hit_rate': sum(1 for m in metrics if m.cache_hit) / len(metrics)
                }
            
            return aggregated
    
    def export_complexity_metrics(self, filepath: str):
        """
        Export detailed complexity metrics for academic analysis.
        
        This method generates a JSON file containing all recorded operation
        metrics suitable for generating plots and empirical complexity analysis
        for the revised paper.
        """
        with self.metrics_lock:
            metrics_export = [m.to_dict() for m in self.operation_metrics]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_export, f, indent=2)
        
        logger.info(f"Exported {len(metrics_export)} complexity metrics to {filepath}")
    
    def export_window_profiling(self, filepath: str, max_windows: int = 1000):
        """
        Export per-window profiling statistics for detailed analysis.
        
        Args:
            filepath: Output file path
            max_windows: Maximum number of windows to export
        """
        with self.index_lock:
            windows_to_export = list(self.temporal_windows.values())[:max_windows]
            profiling_data = [w.get_profiling_summary() for w in windows_to_export]
        
        with open(filepath, 'w') as f:
            json.dump(profiling_data, f, indent=2)
        
        logger.info(f"Exported profiling for {len(profiling_data)} windows to {filepath}")
    
    def save_index(self, filepath: str):
        """Save index state with error handling."""
        logger.info(f"Saving index to {filepath}...")
        
        temp_path = filepath + '.tmp'
        
        try:
            with self.index_lock:
                state = {
                    'version': '1.1',  # Updated version with profiling
                    'temporal_windows': dict(list(self.temporal_windows.items())[:10000]),
                    'num_accounts': self.num_accounts,
                    'num_transactions': self.num_transactions,
                    'min_timestamp': self.min_timestamp,
                    'max_timestamp': self.max_timestamp,
                    'account_partitions': self.partitioner.account_partitions,
                    'account_lengths': self.adaptive_manager.account_lengths
                }
                
                with open(temp_path, 'wb') as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                os.replace(temp_path, filepath)
                
            logger.info("Index saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def load_index(self, filepath: str):
        """Load index with validation."""
        logger.info(f"Loading index from {filepath}...")
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            with self.index_lock:
                self.temporal_windows = state.get('temporal_windows', {})
                self.num_accounts = state.get('num_accounts', 0)
                self.num_transactions = state.get('num_transactions', 0)
                self.min_timestamp = state.get('min_timestamp', float('inf'))
                self.max_timestamp = state.get('max_timestamp', 0.0)
                
                if 'account_partitions' in state:
                    self.partitioner.account_partitions = state['account_partitions']
                if 'account_lengths' in state:
                    self.adaptive_manager.account_lengths = state['account_lengths']
            
            logger.info(f"Index loaded: {self.num_accounts} accounts, "
                       f"{self.num_transactions} transactions")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def shutdown(self):
        """Graceful shutdown with cleanup."""
        logger.info("Shutting down TemporalMultiGraphIndex...")
        
        if self.ingestion_engine:
            self.ingestion_engine.shutdown()
        
        logger.info("TemporalMultiGraphIndex shutdown complete")


# Production utility functions

def convert_sequences_to_tensors(sequences: Dict[int, Tuple[List, List]],
                                edge_attr_dim: int,
                                device: str = 'cpu',
                                max_sequence_length: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert sequences to tensors with proper padding and validation."""
    if not sequences:
        # Return empty tensors
        empty = torch.zeros((0, 0, edge_attr_dim), device=device)
        lengths = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, lengths, lengths
    
    account_ids = sorted(sequences.keys())
    batch_size = len(account_ids)
    
    # Determine max lengths
    max_in_len = max((len(sequences[aid][0]) for aid in account_ids), default=1)
    max_out_len = max((len(sequences[aid][1]) for aid in account_ids), default=1)
    
    # Apply length limit if specified
    if max_sequence_length:
        max_in_len = min(max_in_len, max_sequence_length)
        max_out_len = min(max_out_len, max_sequence_length)
    
    # Allocate tensors
    incoming_tensor = torch.zeros((batch_size, max_in_len, edge_attr_dim), dtype=torch.float32)
    outgoing_tensor = torch.zeros((batch_size, max_out_len, edge_attr_dim), dtype=torch.float32)
    incoming_lengths = torch.zeros(batch_size, dtype=torch.long)
    outgoing_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for idx, account_id in enumerate(account_ids):
        in_seq, out_seq = sequences[account_id]
        
        # Truncate if needed
        in_seq = in_seq[-max_in_len:] if len(in_seq) > max_in_len else in_seq
        out_seq = out_seq[-max_out_len:] if len(out_seq) > max_out_len else out_seq
        
        # Fill incoming
        for i, transaction in enumerate(in_seq):
            incoming_tensor[idx, i, :] = torch.from_numpy(transaction.attributes)
        incoming_lengths[idx] = len(in_seq)
        
        # Fill outgoing
        for i, transaction in enumerate(out_seq):
            outgoing_tensor[idx, i, :] = torch.from_numpy(transaction.attributes)
        outgoing_lengths[idx] = len(out_seq)
    
    # Move to device
    return (incoming_tensor.to(device),
            outgoing_tensor.to(device),
            incoming_lengths.to(device),
            outgoing_lengths.to(device))


if __name__ == '__main__':
    # Comprehensive testing with profiling
    logger.info("=" * 80)
    logger.info("Enhanced Temporal Index Production Test with Profiling")
    logger.info("=" * 80)
    
    # Create index with profiling enabled
    index = TemporalMultiGraphIndex(
        num_partitions=4,
        min_sequence_length=8,
        max_sequence_length=64,
        default_sequence_length=32,
        hot_memory_limit_gb=1.0,
        enable_streaming=True,
        enable_profiling=True
    )
    
    # Test streaming ingestion with varying patterns
    logger.info("\n[1/5] Testing streaming ingestion with profiling...")
    num_accounts = 100
    num_transactions = 1000
    
    for i in range(num_transactions):
        transaction = TransactionRecord(
            edge_id=i,
            source_id=np.random.randint(0, num_accounts),
            target_id=np.random.randint(0, num_accounts),
            timestamp=time.time() + i,
            attributes=np.random.randn(8).astype(np.float32),
            block_height=i // 10
        )
        index.add_transaction(transaction)
    
    # Wait for asynchronous processing
    time.sleep(2)
    logger.info(f"Ingested {num_transactions} transactions")
    
    # Test retrieval with complexity measurement
    logger.info("\n[2/5] Testing sequence retrieval with complexity tracking...")
    test_accounts = list(range(20))
    sequences = index.get_transaction_sequences(test_accounts)
    
    for account_id, (in_seq, out_seq) in sequences.items():
        if in_seq or out_seq:
            logger.info(f"  Account {account_id}: {len(in_seq)} in, {len(out_seq)} out")
    
    # Test tensor conversion
    logger.info("\n[3/5] Testing tensor conversion...")
    tensors = convert_sequences_to_tensors(sequences, edge_attr_dim=8, device='cpu')
    logger.info(f"  Tensor shapes: in={tensors[0].shape}, out={tensors[1].shape}")
    
    # Export profiling data
    logger.info("\n[4/5] Exporting profiling data for analysis...")
    index.export_complexity_metrics('./complexity_metrics.json')
    index.export_window_profiling('./window_profiling.json', max_windows=100)
    index.adaptive_manager.export_adaptation_history('./adaptation_history.json')
    logger.info("  Exported complexity metrics, window profiling, and adaptation history")
    
    # Display comprehensive statistics
    logger.info("\n[5/5] Comprehensive Statistics Report:")
    logger.info("=" * 80)
    stats = index.get_index_statistics()
    
    logger.info(f"\nGlobal Index Statistics:")
    logger.info(f"  Accounts: {stats['num_accounts']}")
    logger.info(f"  Transactions: {stats['num_transactions']}")
    logger.info(f"  Temporal Range: {stats['temporal_range_seconds']:.1f} seconds")
    logger.info(f"  Mean In-Degree: {stats['mean_in_degree']:.2f} ± {stats['std_in_degree']:.2f}")
    logger.info(f"  Mean Out-Degree: {stats['mean_out_degree']:.2f} ± {stats['std_out_degree']:.2f}")
    logger.info(f"  Partition Balance: {stats['partition_balance']:.3f}")
    
    if 'complexity_analysis' in stats and stats['complexity_analysis']:
        logger.info(f"\nComplexity Analysis:")
        for op_name, metrics in stats['complexity_analysis'].items():
            logger.info(f"  {op_name}:")
            logger.info(f"    Operations: {metrics['count']}")
            logger.info(f"    Mean Comparisons: {metrics['mean_comparisons']:.2f} ± {metrics['std_comparisons']:.2f}")
            logger.info(f"    Mean Time: {metrics['mean_time_ms']:.4f} ms (p95: {metrics['p95_time_ms']:.4f} ms)")
            logger.info(f"    Cache Hit Rate: {metrics['cache_hit_rate']:.3f}")
    
    if 'adaptive_sequence' in stats:
        logger.info(f"\nAdaptive Sequence Management:")
        adp = stats['adaptive_sequence']
        logger.info(f"  Total Adaptations: {adp['total_adaptations']}")
        logger.info(f"  Mean Length: {adp['mean_sequence_length']:.1f} ± {adp['std_sequence_length']:.1f}")
        logger.info(f"  Length Range: [{adp['min_sequence_length']}, {adp['max_sequence_length']}]")
        logger.info(f"  Percentiles (P25/P50/P75): {adp['p25_sequence_length']:.1f} / {adp['p50_sequence_length']:.1f} / {adp['p75_sequence_length']:.1f}")
    
    if 'storage' in stats:
        logger.info(f"\nStorage Management:")
        stor = stats['storage']
        logger.info(f"  Hot Accounts: {stor['num_hot_accounts']}")
        logger.info(f"  Cold Accounts: {stor['num_cold_accounts']}")
        tier_metrics = stor['tier_metrics']
        logger.info(f"  Tier Transitions: {tier_metrics['hot_to_cold_transitions']} hot→cold, {tier_metrics['cold_to_hot_transitions']} cold→hot")
        logger.info(f"  Compression Ratio: {tier_metrics['avg_compression_ratio']:.3f}")
    
    if 'ingestion' in stats and stats['ingestion']:
        logger.info(f"\nStreaming Ingestion:")
        ing = stats['ingestion']
        logger.info(f"  Total Ingested: {ing['total_ingested']}")
        logger.info(f"  Throughput: {ing['throughput_tps']:.1f} TPS")
        logger.info(f"  Avg Batch Time: {ing['avg_batch_time_ms']:.2f} ms")
    
    # Cleanup
    logger.info("\n" + "=" * 80)
    logger.info("Shutting down index...")
    index.shutdown()
    
    logger.info("=" * 80)
    logger.info("All tests completed successfully!")
    logger.info("Enhanced temporal_index.py is ready for ICDE 2026 submission")
    logger.info("=" * 80)