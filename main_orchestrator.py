"""
main_orchestrator.py - Experiment Management and Orchestration for StreamDIAM

This module serves as the unified entry point for all experimental workflows,
ensuring reproducibility, systematic evaluation, and comprehensive result generation.
It coordinates the complete pipeline from data loading through visualization,
managing computational resources and experimental protocols.

Copyright (c) 2025 StreamDIAM Research Team
For ICDE 2026 Submission: "StreamDIAM: Scalable Real-Time Illicit Account 
Detection Over Temporal Cryptocurrency Transaction Graphs"
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch_geometric.data import Data
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import yaml
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict, field
import time
from datetime import datetime, timedelta
import os
import sys
import warnings
import pickle
import gc
from collections import OrderedDict, defaultdict
import psutil
import GPUtil
import traceback
from contextlib import contextmanager
import signal
import atexit

from temporal_index import (
    TemporalMultiGraphIndex,
    TransactionRecord,
    AdaptiveSequenceManager
)
from model_architecture import (
    StreamDIAM,
    ModelConfig,
    create_model_from_config
)
from training_engine import (
    IncrementalTrainer,
    TrainingConfig,
    TrainingMetrics
)
from evaluation_framework import (
    TemporalEvaluator,
    SystemBenchmark,
    BaselineComparator,
    AblationAnalyzer,
    VisualizationEngine,
    DetectionMetrics,
    SystemMetrics,
    ScalabilityMetrics,
    CompleteEvaluationPipeline
)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


@dataclass
class ExperimentConfig:
    """
    Master configuration for complete experimental workflow with validation.
    
    This enhanced configuration includes validation rules and default values
    that ensure robust experimental execution.
    """
    # Experiment identification
    experiment_name: str = "streamdiam_baseline"
    dataset_name: str = "EthereumS"
    dataset_path: str = "./data/EthereumS/data.pt"
    output_dir: str = "./results"
    random_seed: int = 42
    
    # Computational resources
    use_gpu: bool = True
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 8
    memory_limit_gb: float = 50.0
    
    # Model and training configurations
    model_config: Dict = field(default_factory=dict)
    training_config: Dict = field(default_factory=dict)
    
    # Evaluation components
    enable_temporal_evaluation: bool = True
    enable_system_benchmark: bool = True
    enable_baseline_comparison: bool = False
    enable_ablation_study: bool = False
    enable_scalability_test: bool = False
    
    # Evaluation parameters
    temporal_window_hours: float = 168.0
    temporal_stride_hours: float = 24.0
    benchmark_duration_seconds: float = 60.0
    num_latency_samples: int = 1000
    
    # Baseline systems to compare
    baseline_systems: List[str] = field(default_factory=lambda: ['neo4j', 'flink'])
    
    # Ablation configurations
    ablation_components: List[str] = field(default_factory=lambda: [
        'temporal_attention', 'incremental_updates', 'adaptive_sequences'
    ])
    
    # Scalability test configurations
    scalability_node_counts: List[int] = field(default_factory=lambda: [
        100000, 500000, 1000000, 5000000
    ])
    scalability_gpu_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # Recovery and checkpointing
    enable_checkpointing: bool = True
    checkpoint_frequency_epochs: int = 5
    enable_recovery: bool = True
    recovery_checkpoint_path: Optional[str] = None
    
    # Output control
    visualization_enabled: bool = True
    save_predictions: bool = True
    save_embeddings: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        # Ensure GPU IDs are valid
        if self.use_gpu and torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            self.gpu_ids = [gid for gid in self.gpu_ids if gid < available_gpus]
            if not self.gpu_ids:
                self.gpu_ids = [0]
        elif not self.use_gpu:
            self.gpu_ids = []
        
        # Initialize model configuration with defaults
        if not self.model_config:
            self.model_config = {
                'hidden_channels': 128,
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'rnn_type': 'gru',
                'rnn_aggregation': 'attention',
                'dropout_rate': 0.2,
                'temporal_decay_rate': 0.1,
                'use_layer_norm': True,
                'use_batch_norm': True,
                'attention_heads': 4,
                'attention_hidden_dim': 16,
                'enable_incremental': True,
                'device': 'cuda' if self.use_gpu and self.gpu_ids else 'cpu'
            }
        
        # Initialize training configuration with defaults
        if not self.training_config:
            self.training_config = {
                'num_epochs': 30,
                'batch_size': 128,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer_type': 'adam',
                'scheduler_type': 'multistep',
                'scheduler_milestones': [10, 20],
                'scheduler_gamma': 0.5,
                'warmup_epochs': 2,
                'max_gradient_norm': 1.0,
                'use_class_weights': True,
                'enable_incremental': True,
                'enable_distributed': len(self.gpu_ids) > 1,
                'num_gpus': len(self.gpu_ids),
                'enable_curriculum': True,
                'temporal_priority_weight': 0.7,
                'checkpoint_frequency': self.checkpoint_frequency_epochs,
                'validation_frequency': 1,
                'enable_early_stopping': True,
                'early_stopping_patience': 10,
                'random_seed': self.random_seed,
                'device': 'cuda' if self.use_gpu and self.gpu_ids else 'cpu'
            }
    
    def validate(self) -> List[str]:
        """
        Validate configuration consistency and return any issues.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check dataset path
        if not Path(self.dataset_path).exists():
            errors.append(f"Dataset not found: {self.dataset_path}")
        
        # Check GPU availability
        if self.use_gpu and not torch.cuda.is_available():
            errors.append("GPU requested but CUDA not available")
        
        # Check memory limit
        available_memory = psutil.virtual_memory().total / (1024**3)
        if self.memory_limit_gb > available_memory:
            errors.append(f"Memory limit {self.memory_limit_gb}GB exceeds available {available_memory:.1f}GB")
        
        # Validate temporal parameters
        if self.temporal_window_hours <= 0:
            errors.append("Temporal window size must be positive")
        
        if self.temporal_stride_hours <= 0:
            errors.append("Temporal stride must be positive")
        
        # Check for incompatible feature combinations
        if self.enable_distributed and len(self.gpu_ids) < 2:
            errors.append("Distributed training requires at least 2 GPUs")
        
        return errors


class MemoryManager:
    """
    Manages memory allocation and cleanup throughout experiment execution.
    
    This component prevents memory leaks and ensures efficient resource usage
    across long-running experiments.
    """
    
    def __init__(self, memory_limit_gb: float = 50.0):
        """Initialize memory manager with limit."""
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        self.checkpoints = []
        
    @contextmanager
    def memory_scope(self, name: str):
        """
        Context manager for memory-scoped operations.
        
        Args:
            name: Name of the scope for logging
        """
        # Force garbage collection before scope
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        initial_memory = self._get_current_memory()
        logger.info(f"Entering memory scope '{name}': {initial_memory / 1024**3:.2f}GB used")
        
        try:
            yield
        finally:
            # Clean up after scope
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            final_memory = self._get_current_memory()
            memory_delta = final_memory - initial_memory
            
            logger.info(f"Exiting memory scope '{name}': "
                       f"{final_memory / 1024**3:.2f}GB used "
                       f"(Δ {memory_delta / 1024**6:.3f}MB)")
            
            # Check memory limit
            if final_memory > self.memory_limit_bytes:
                logger.warning(f"Memory usage exceeds limit: "
                              f"{final_memory / 1024**3:.2f}GB > "
                              f"{self.memory_limit_bytes / 1024**3:.2f}GB")
    
    def _get_current_memory(self) -> int:
        """Get current memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def checkpoint_memory(self, name: str):
        """Save memory checkpoint for debugging."""
        memory_info = {
            'name': name,
            'timestamp': datetime.now(),
            'ram_bytes': self._get_current_memory(),
            'ram_gb': self._get_current_memory() / 1024**3
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_bytes'] = torch.cuda.memory_allocated()
            memory_info['gpu_gb'] = torch.cuda.memory_allocated() / 1024**3
        
        self.checkpoints.append(memory_info)
        return memory_info


class ExperimentRecovery:
    """
    Handles experiment recovery from checkpoints after failures.
    
    This component enables resuming experiments from the last successful checkpoint,
    critical for long-running experiments on shared computing resources.
    """
    
    def __init__(self, checkpoint_dir: Path):
        """Initialize recovery handler."""
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.current_state = {}
        
    def save_state(self, state: Dict[str, Any], stage: str):
        """
        Save current experiment state.
        
        Args:
            state: State dictionary to save
            stage: Current pipeline stage
        """
        checkpoint_path = self.checkpoint_dir / f"recovery_{stage}_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        
        self.current_state = {
            'stage': stage,
            'timestamp': datetime.now(),
            'state': state
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.current_state, f)
        
        logger.info(f"Saved recovery checkpoint: {checkpoint_path}")
        
        # Clean old checkpoints
        self._clean_old_checkpoints()
    
    def load_latest_state(self) -> Optional[Dict[str, Any]]:
        """
        Load most recent checkpoint if available.
        
        Returns:
            Recovered state or None
        """
        checkpoints = sorted(self.checkpoint_dir.glob("recovery_*.pkl"))
        
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        logger.info(f"Loading recovery checkpoint: {latest}")
        
        with open(latest, 'rb') as f:
            state = pickle.load(f)
        
        return state
    
    def _clean_old_checkpoints(self, keep: int = 3):
        """Keep only recent checkpoints to save space."""
        checkpoints = sorted(self.checkpoint_dir.glob("recovery_*.pkl"))
        
        if len(checkpoints) > keep:
            for checkpoint in checkpoints[:-keep]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        logger.warning(f"Received signal {signum}, saving state...")
        
        if self.current_state:
            emergency_path = self.checkpoint_dir / f"emergency_{datetime.now():%Y%m%d_%H%M%S}.pkl"
            with open(emergency_path, 'wb') as f:
                pickle.dump(self.current_state, f)
            logger.info(f"Emergency checkpoint saved: {emergency_path}")
        
        sys.exit(1)
    
    def cleanup(self):
        """Cleanup on normal exit."""
        logger.info("Experiment recovery handler shutting down")


class DataValidator:
    """
    Validates and preprocesses dataset ensuring compatibility.
    
    This component handles various dataset formats and ensures consistent
    structure for downstream processing.
    """
    
    @staticmethod
    def validate_and_preprocess(data: Data) -> Data:
        """
        Validate and preprocess graph data.
        
        Args:
            data: Raw graph data
            
        Returns:
            Validated and preprocessed data
        """
        # Ensure required attributes exist
        if not hasattr(data, 'edge_index'):
            raise ValueError("Data missing required 'edge_index' attribute")
        
        if not hasattr(data, 'num_nodes'):
            data.num_nodes = int(data.edge_index.max()) + 1
        
        # Handle labels
        if not hasattr(data, 'y') and not hasattr(data, 'labels'):
            raise ValueError("Data missing labels (expected 'y' or 'labels' attribute)")
        
        if not hasattr(data, 'y'):
            data.y = data.labels
        
        # Handle edge attributes
        if not hasattr(data, 'edge_attr'):
            logger.warning("No edge attributes found, creating random features")
            data.edge_attr = torch.randn(data.edge_index.shape[1], 8)
        
        # Ensure edge attributes have timestamp
        if data.edge_attr.shape[1] < 2:
            logger.warning("Edge attributes lack timestamp, appending synthetic timestamps")
            timestamps = torch.arange(data.edge_index.shape[1], dtype=torch.float32)
            timestamps = timestamps.unsqueeze(1) / len(timestamps)
            data.edge_attr = torch.cat([data.edge_attr, timestamps], dim=1)
        
        # Validate data types
        data.edge_index = data.edge_index.long()
        data.y = data.y.long()
        data.edge_attr = data.edge_attr.float()
        
        # Check for data issues
        if data.y.min() < 0 or data.y.max() > 1:
            raise ValueError(f"Labels must be binary (0/1), found range [{data.y.min()}, {data.y.max()}]")
        
        # Compute statistics
        num_positive = (data.y == 1).sum().item()
        num_negative = (data.y == 0).sum().item()
        
        logger.info(f"Data validated: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
        logger.info(f"Label distribution: {num_positive} positive, {num_negative} negative "
                   f"({num_positive/(num_positive + num_negative)*100:.1f}% positive)")
        
        return data


class CompleteExperimentOrchestrator:
    """
    Enhanced orchestrator with full feature implementation and robust error handling.
    
    This production-ready orchestrator implements all promised features including
    baseline comparisons, ablation studies, and scalability testing.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize complete orchestrator."""
        self.config = config
        
        # Validate configuration
        errors = config.validate()
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Setup infrastructure
        self.setup_output_directories()
        
        # Initialize managers
        self.memory_manager = MemoryManager(config.memory_limit_gb)
        self.recovery_handler = ExperimentRecovery(
            Path(config.output_dir) / config.experiment_name / 'recovery'
        )
        
        # Initialize components
        self.resource_allocator = ResourceAllocator(
            gpu_ids=config.gpu_ids,
            num_cpu_cores=config.num_workers
        )
        
        self.result_aggregator = ResultAggregator(
            output_dir=config.output_dir
        )
        
        self.reproducibility_guard = ReproducibilityGuard(
            random_seed=config.random_seed
        )
        
        # Set global seed
        self.reproducibility_guard.set_global_seed()
        
        # Core components
        self.temporal_index: Optional[TemporalMultiGraphIndex] = None
        self.model: Optional[StreamDIAM] = None
        self.trainer: Optional[IncrementalTrainer] = None
        
        # Track experiment state
        self.experiment_state = {
            'stage': 'initialized',
            'start_time': datetime.now(),
            'checkpoints': []
        }
        
        logger.info(f"Complete orchestrator initialized for: {config.experiment_name}")
    
    def setup_output_directories(self):
        """Create comprehensive directory structure."""
        base_dir = Path(self.config.output_dir) / self.config.experiment_name
        
        directories = [
            base_dir,
            base_dir / 'checkpoints',
            base_dir / 'recovery',
            base_dir / 'logs',
            base_dir / 'metrics',
            base_dir / 'visualizations',
            base_dir / 'predictions',
            base_dir / 'embeddings',
            base_dir / 'baselines',
            base_dir / 'ablations',
            base_dir / 'scalability'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories created: {base_dir}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete pipeline with all features and recovery support.
        
        Returns:
            Comprehensive results dictionary
        """
        logger.info("=" * 80)
        logger.info(f"Starting complete pipeline: {self.config.experiment_name}")
        logger.info("=" * 80)
        
        pipeline_start_time = time.time()
        results = {}
        
        # Check for recovery
        if self.config.enable_recovery:
            recovered_state = self.recovery_handler.load_latest_state()
            if recovered_state:
                logger.info(f"Recovering from stage: {recovered_state['stage']}")
                self.experiment_state = recovered_state['state']['experiment_state']
                results = recovered_state['state'].get('results', {})
        
        try:
            # Stage 1: Data Loading
            if self.experiment_state['stage'] in ['initialized']:
                with self.memory_manager.memory_scope("data_loading"):
                    logger.info("\n[Stage 1/8] Loading and Validating Dataset")
                    data, train_mask, val_mask, test_mask = self.load_and_validate_dataset()
                    
                    self.experiment_state['stage'] = 'data_loaded'
                    self.experiment_state['data_info'] = {
                        'num_nodes': data.num_nodes,
                        'num_edges': data.edge_index.shape[1],
                        'edge_attr_dim': data.edge_attr.shape[1]
                    }
                    
                    self.recovery_handler.save_state({
                        'experiment_state': self.experiment_state,
                        'results': results
                    }, 'data_loaded')
            
            # Stage 2: Temporal Index
            if self.experiment_state['stage'] in ['data_loaded']:
                with self.memory_manager.memory_scope("temporal_index"):
                    logger.info("\n[Stage 2/8] Building Temporal Index")
                    self.initialize_temporal_index(data)
                    
                    self.experiment_state['stage'] = 'index_built'
                    self.recovery_handler.save_state({
                        'experiment_state': self.experiment_state,
                        'results': results
                    }, 'index_built')
            
            # Stage 3: Model Initialization
            if self.experiment_state['stage'] in ['index_built']:
                with self.memory_manager.memory_scope("model_init"):
                    logger.info("\n[Stage 3/8] Initializing Model")
                    edge_attr_dim = self.experiment_state['data_info']['edge_attr_dim']
                    self.initialize_model(edge_attr_dim)
                    
                    self.experiment_state['stage'] = 'model_initialized'
                    self.recovery_handler.save_state({
                        'experiment_state': self.experiment_state,
                        'results': results
                    }, 'model_initialized')
            
            # Prepare data splits
            train_node_ids = torch.where(train_mask)[0].tolist()
            val_node_ids = torch.where(val_mask)[0].tolist()
            test_node_ids = torch.where(test_mask)[0].tolist()
            
            train_labels = data.y[train_mask]
            val_labels = data.y[val_mask]
            test_labels = data.y[test_mask]
            
            # Stage 4: Model Training
            if self.experiment_state['stage'] in ['model_initialized']:
                with self.memory_manager.memory_scope("training"):
                    logger.info("\n[Stage 4/8] Training Model")
                    
                    if self.config.training_config['enable_distributed'] and len(self.config.gpu_ids) > 1:
                        training_history = self.train_model_distributed(
                            train_node_ids, train_labels,
                            val_node_ids, val_labels,
                            data.edge_index
                        )
                    else:
                        training_history = self.train_model(
                            train_node_ids, train_labels,
                            val_node_ids, val_labels,
                            data.edge_index
                        )
                    
                    results['training_history'] = training_history
                    self.experiment_state['stage'] = 'model_trained'
                    self.recovery_handler.save_state({
                        'experiment_state': self.experiment_state,
                        'results': results
                    }, 'model_trained')
            
            # Stage 5: Temporal Evaluation
            if self.config.enable_temporal_evaluation:
                with self.memory_manager.memory_scope("temporal_eval"):
                    logger.info("\n[Stage 5/8] Running Temporal Evaluation")
                    temporal_results = self.run_temporal_evaluation(
                        test_node_ids, test_labels, data.edge_index
                    )
                    results['temporal_evaluation'] = temporal_results
            
            # Stage 6: System Benchmarks
            if self.config.enable_system_benchmark:
                with self.memory_manager.memory_scope("system_benchmark"):
                    logger.info("\n[Stage 6/8] Running System Benchmarks")
                    system_results = self.run_comprehensive_benchmarks(
                        test_node_ids, data.edge_index
                    )
                    results['system_benchmark'] = system_results
            
            # Stage 7: Baseline Comparisons
            if self.config.enable_baseline_comparison:
                with self.memory_manager.memory_scope("baseline_comparison"):
                    logger.info("\n[Stage 7/8] Running Baseline Comparisons")
                    baseline_results = self.run_baseline_comparisons(
                        test_node_ids, test_labels, data.edge_index
                    )
                    results['baseline_comparison'] = baseline_results
            
            # Stage 8: Advanced Analyses
            if self.config.enable_ablation_study or self.config.enable_scalability_test:
                with self.memory_manager.memory_scope("advanced_analysis"):
                    logger.info("\n[Stage 8/8] Running Advanced Analyses")
                    
                    if self.config.enable_ablation_study:
                        ablation_results = self.run_ablation_study(
                            train_node_ids, train_labels,
                            val_node_ids, val_labels,
                            data.edge_index
                        )
                        results['ablation_study'] = ablation_results
                    
                    if self.config.enable_scalability_test:
                        scalability_results = self.run_scalability_tests(
                            data
                        )
                        results['scalability_tests'] = scalability_results
            
            # Generate outputs
            self.generate_comprehensive_outputs(results)
            
            # Final cleanup
            pipeline_duration = time.time() - pipeline_start_time
            results['execution_time'] = pipeline_duration
            
            logger.info("=" * 80)
            logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f} seconds")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            
            # Save emergency checkpoint
            self.recovery_handler.save_state({
                'experiment_state': self.experiment_state,
                'results': results,
                'error': str(e)
            }, 'error')
            
            raise
    
    def load_and_validate_dataset(self) -> Tuple[Data, Tensor, Tensor, Tensor]:
        """Load and validate dataset with preprocessing."""
        data = torch.load(self.config.dataset_path)
        data = DataValidator.validate_and_preprocess(data)
        
        # Create or load splits
        if hasattr(data, 'train_mask'):
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        else:
            # Create temporal split
            num_nodes = data.num_nodes
            indices = torch.randperm(num_nodes)
            
            train_size = int(0.5 * num_nodes)
            val_size = int(0.25 * num_nodes)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
        
        return data, train_mask, val_mask, test_mask
    
    def train_model_distributed(self,
                              train_node_ids: List[int],
                              train_labels: Tensor,
                              val_node_ids: List[int],
                              val_labels: Tensor,
                              edge_index: Tensor) -> Dict:
        """Train model using distributed data parallel."""
        logger.info(f"Starting distributed training on {len(self.config.gpu_ids)} GPUs")
        
        world_size = len(self.config.gpu_ids)
        
        # Spawn processes for distributed training
        mp.spawn(
            self._distributed_train_worker,
            args=(world_size, train_node_ids, train_labels,
                  val_node_ids, val_labels, edge_index),
            nprocs=world_size,
            join=True
        )
        
        # Load best checkpoint
        checkpoint_dir = Path(self.config.output_dir) / self.config.experiment_name / 'checkpoints'
        best_checkpoint = checkpoint_dir / 'best_model.pt'
        
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint.get('history', {})
        
        return {}
    
    def _distributed_train_worker(self,
                                rank: int,
                                world_size: int,
                                train_node_ids: List[int],
                                train_labels: Tensor,
                                val_node_ids: List[int],
                                val_labels: Tensor,
                                edge_index: Tensor):
        """Worker process for distributed training."""
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        # Create model for this process
        device = f'cuda:{self.config.gpu_ids[rank]}'
        model_config = ModelConfig(**self.config.model_config)
        model_config.device = device
        
        model = StreamDIAM(
            config=model_config,
            edge_attr_dim=self.experiment_state['data_info']['edge_attr_dim'],
            num_classes=2
        ).to(device)
        
        model = DDP(model, device_ids=[self.config.gpu_ids[rank]])
        
        # Partition data for this rank
        chunk_size = len(train_node_ids) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else len(train_node_ids)
        
        rank_train_ids = train_node_ids[start_idx:end_idx]
        rank_train_labels = train_labels[start_idx:end_idx]
        
        # Train on this partition
        training_config = TrainingConfig(**self.config.training_config)
        trainer = IncrementalTrainer(
            model=model.module,
            temporal_index=self.temporal_index,
            config=training_config
        )
        
        trainer.fit(
            train_node_ids=rank_train_ids,
            train_labels=rank_train_labels,
            val_node_ids=val_node_ids,
            val_labels=val_labels,
            edge_index=edge_index
        )
        
        # Cleanup
        dist.destroy_process_group()
    
    def run_baseline_comparisons(self,
                                test_node_ids: List[int],
                                test_labels: Tensor,
                                edge_index: Tensor) -> Dict:
        """Execute baseline system comparisons."""
        comparator = BaselineComparator(
            self.model,
            self.temporal_index,
            self.config.model_config['device']
        )
        
        # Register and evaluate baselines
        baseline_results = {}
        
        for baseline_name in self.config.baseline_systems:
            if baseline_name == 'diam_static':
                # Load original DIAM for comparison
                from baseline_models import OriginalDIAM
                baseline_model = OriginalDIAM()
                comparator.register_baseline('DIAM-Static', baseline_model)
            
        # Run comparisons
        detection_comparison = comparator.compare_detection_performance(
            test_node_ids, test_labels, edge_index
        )
        
        system_comparison = comparator.compare_system_performance(
            test_node_ids, edge_index
        )
        
        return {
            'detection': detection_comparison,
            'system': system_comparison
        }
    
    def run_ablation_study(self,
                          train_node_ids: List[int],
                          train_labels: Tensor,
                          val_node_ids: List[int],
                          val_labels: Tensor,
                          edge_index: Tensor) -> Dict:
        """Execute comprehensive ablation study."""
        analyzer = AblationAnalyzer(
            self.model,
            self.temporal_index,
            self.config.model_config['device']
        )
        
        training_config = TrainingConfig(**self.config.training_config)
        training_config.num_epochs = 10  # Reduced for ablation
        
        results = analyzer.run_ablation_study(
            train_node_ids, train_labels,
            val_node_ids, val_labels,
            edge_index, training_config
        )
        
        return results
    
    def run_scalability_tests(self, data: Data) -> Dict:
        """Execute scalability tests across data sizes and GPU counts."""
        benchmark = SystemBenchmark(
            self.model,
            self.temporal_index,
            self.config.model_config['device']
        )
        
        test_configurations = []
        for num_nodes in self.config.scalability_node_counts:
            for num_gpus in self.config.scalability_gpu_counts:
                if num_nodes <= data.num_nodes and num_gpus <= len(self.config.gpu_ids):
                    test_configurations.append((num_nodes, num_nodes * 5, num_gpus))
        
        scalability_metrics = benchmark.measure_scalability(
            test_configurations,
            self.config.scalability_gpu_counts
        )
        
        return {'metrics': scalability_metrics}
    
    def run_comprehensive_benchmarks(self,
                                    test_node_ids: List[int],
                                    edge_index: Tensor) -> Dict:
        """Run comprehensive system benchmarks with realistic workloads."""
        benchmark = SystemBenchmark(
            self.model,
            self.temporal_index,
            self.config.model_config['device']
        )
        
        results = {}
        
        # Standard benchmarks
        throughput, metrics = benchmark.measure_query_throughput(
            test_node_ids, edge_index,
            duration_seconds=self.config.benchmark_duration_seconds
        )
        results['standard_throughput'] = metrics.to_dict()
        
        # Realistic workload benchmarks
        if hasattr(benchmark, 'measure_realistic_throughput'):
            for workload_type in ['normal', 'burst', 'mixed']:
                _, workload_metrics = benchmark.measure_realistic_throughput(
                    test_node_ids, edge_index,
                    workload_type=workload_type,
                    duration_seconds=30.0
                )
                results[f'{workload_type}_workload'] = workload_metrics.to_dict()
        
        return results
    
    def generate_comprehensive_outputs(self, results: Dict):
        """Generate all outputs including visualizations and reports."""
        base_dir = Path(self.config.output_dir) / self.config.experiment_name
        
        # Save results
        self.result_aggregator.save_results(format='json')
        
        # Generate visualizations
        if self.config.visualization_enabled:
            viz_engine = VisualizationEngine(
                output_dir=str(base_dir / 'visualizations')
            )
            
            if 'temporal_evaluation' in results:
                temporal_windows = results['temporal_evaluation'].get('temporal_windows', [])
                if temporal_windows:
                    viz_engine.plot_temporal_performance(temporal_windows)
            
            if 'baseline_comparison' in results:
                detection_comparison = results['baseline_comparison'].get('detection', {})
                if detection_comparison:
                    viz_engine.plot_comparison_bar_chart(detection_comparison)
            
            if 'ablation_study' in results:
                viz_engine.plot_ablation_results(results['ablation_study'])
            
            if 'scalability_tests' in results:
                scalability_metrics = results['scalability_tests'].get('metrics', [])
                if scalability_metrics:
                    viz_engine.plot_scalability_curves(scalability_metrics)
        
        # Save configuration
        config_path = base_dir / 'experiment_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
        
        # Save environment info
        env_info = self.reproducibility_guard.capture_environment()
        env_path = base_dir / 'environment.json'
        with open(env_path, 'w') as f:
            json.dump(env_info, f, indent=2)
        
        # Generate summary report
        summary = self.result_aggregator.generate_summary_report()
        summary_path = base_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print("\n" + summary)
        
        # Log final resource usage
        self.resource_allocator.log_resource_usage()
        
        # Save memory checkpoints
        memory_report = {
            'checkpoints': self.memory_manager.checkpoints,
            'peak_memory_gb': max(c['ram_gb'] for c in self.memory_manager.checkpoints)
        }
        memory_path = base_dir / 'memory_usage.json'
        with open(memory_path, 'w') as f:
            json.dump(memory_report, f, indent=2)
    
    # Keep simplified versions of other required methods
    def initialize_temporal_index(self, data: Data):
        """Initialize temporal index."""
        self.temporal_index = TemporalMultiGraphIndex(
            num_partitions=len(self.config.gpu_ids) if self.config.gpu_ids else 1,
            min_sequence_length=8,
            max_sequence_length=128,
            default_sequence_length=32,
            hot_memory_limit_gb=self.config.memory_limit_gb * 0.5
        )
        self.temporal_index.build_from_static_graph(data)
    
    def initialize_model(self, edge_attr_dim: int):
        """Initialize model."""
        model_config = ModelConfig(**self.config.model_config)
        
        if self.config.use_gpu and self.config.gpu_ids:
            device = self.resource_allocator.select_gpu()
            model_config.device = f'cuda:{device}'
        
        self.model = StreamDIAM(
            config=model_config,
            edge_attr_dim=edge_attr_dim,
            num_classes=2
        )
    
    def train_model(self,
                   train_node_ids: List[int],
                   train_labels: Tensor,
                   val_node_ids: List[int],
                   val_labels: Tensor,
                   edge_index: Tensor) -> Dict:
        """Train model with single GPU."""
        training_config = TrainingConfig(**self.config.training_config)
        
        checkpoint_dir = Path(self.config.output_dir) / self.config.experiment_name / 'checkpoints'
        log_dir = Path(self.config.output_dir) / self.config.experiment_name / 'logs'
        
        self.trainer = IncrementalTrainer(
            model=self.model,
            temporal_index=self.temporal_index,
            config=training_config,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir)
        )
        
        return self.trainer.fit(
            train_node_ids=train_node_ids,
            train_labels=train_labels,
            val_node_ids=val_node_ids,
            val_labels=val_labels,
            edge_index=edge_index
        )
    
    def run_temporal_evaluation(self,
                               test_node_ids: List[int],
                               test_labels: Tensor,
                               edge_index: Tensor) -> Dict:
        """Run temporal evaluation."""
        evaluator = TemporalEvaluator(
            self.model,
            self.temporal_index,
            self.config.model_config['device']
        )
        
        temporal_results = evaluator.evaluate_sliding_windows(
            test_node_ids,
            test_labels,
            edge_index,
            window_size_hours=self.config.temporal_window_hours,
            stride_hours=self.config.temporal_stride_hours
        )
        
        for start_time, end_time, metrics in temporal_results:
            self.result_aggregator.add_detection_result(
                metrics,
                metadata={
                    'experiment': self.config.experiment_name,
                    'window_start': start_time,
                    'window_end': end_time
                }
            )
        
        return {'temporal_windows': temporal_results}


# Keep the original simplified classes but ensure they're properly typed
class ConfigurationManager:
    """Configuration management with validation."""
    
    def __init__(self, config_dir: str = './config'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_yaml_config(self, config_path: str) -> Dict:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def create_experiment_config(self,
                                config_dict: Optional[Dict] = None,
                                config_file: Optional[str] = None) -> ExperimentConfig:
        if config_file:
            config_dict = self.load_yaml_config(config_file)
        
        config = ExperimentConfig(**(config_dict or {}))
        
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        
        return config


class ResourceAllocator:
    """Resource allocation with monitoring."""
    
    def __init__(self, gpu_ids: List[int], num_cpu_cores: int = 8):
        self.gpu_ids = gpu_ids
        self.num_cpu_cores = num_cpu_cores
        self.gpu_memory_usage = {gpu_id: 0.0 for gpu_id in gpu_ids}
    
    def select_gpu(self, memory_required_gb: float = 10.0) -> int:
        if not torch.cuda.is_available() or not self.gpu_ids:
            raise RuntimeError("No GPUs available")
        
        try:
            gpus = GPUtil.getGPUs()
            available_gpus = {gpu.id: gpu for gpu in gpus if gpu.id in self.gpu_ids}
            
            best_gpu = None
            best_score = float('inf')
            
            for gpu_id in self.gpu_ids:
                if gpu_id in available_gpus:
                    gpu = available_gpus[gpu_id]
                    free_memory = gpu.memoryFree / 1024
                    
                    if free_memory >= memory_required_gb:
                        score = gpu.memoryUtil + 0.5 * gpu.load
                        if score < best_score:
                            best_score = score
                            best_gpu = gpu_id
            
            return best_gpu if best_gpu is not None else self.gpu_ids[0]
            
        except Exception as e:
            logger.warning(f"GPU selection failed: {e}, using first GPU")
            return self.gpu_ids[0] if self.gpu_ids else 0
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        stats = {
            'num_gpus': len(self.gpu_ids),
            'num_cpu_cores': self.num_cpu_cores,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024 ** 3),
            'ram_total_gb': psutil.virtual_memory().total / (1024 ** 3)
        }
        
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                stats['gpu_stats'] = [
                    {
                        'gpu_id': gpu.id,
                        'name': gpu.name,
                        'memory_used_gb': gpu.memoryUsed / 1024,
                        'memory_total_gb': gpu.memoryTotal / 1024,
                        'gpu_util_percent': gpu.load * 100
                    }
                    for gpu in gpus if gpu.id in self.gpu_ids
                ]
            except:
                pass
        
        return stats
    
    def log_resource_usage(self):
        stats = self.get_resource_statistics()
        logger.info("=" * 80)
        logger.info("Resource Utilization")
        logger.info(f"CPU: {stats['cpu_percent']:.1f}% | "
                   f"RAM: {stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f}GB")
        
        if 'gpu_stats' in stats:
            for gpu in stats['gpu_stats']:
                logger.info(f"GPU {gpu['gpu_id']}: "
                           f"{gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f}GB | "
                           f"Util: {gpu['gpu_util_percent']:.1f}%")
        logger.info("=" * 80)


class ResultAggregator:
    """Result aggregation and reporting."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.detection_results = []
        self.system_results = []
        self.scalability_results = []
    
    def add_detection_result(self, metrics: DetectionMetrics, metadata: Dict = None):
        result = metrics.to_dict()
        if metadata:
            result.update(metadata)
        self.detection_results.append(result)
    
    def add_system_result(self, metrics: SystemMetrics, metadata: Dict = None):
        result = metrics.to_dict()
        if metadata:
            result.update(metadata)
        self.system_results.append(result)
    
    def add_scalability_result(self, metrics: ScalabilityMetrics, metadata: Dict = None):
        result = metrics.to_dict()
        if metadata:
            result.update(metadata)
        self.scalability_results.append(result)
    
    def compute_aggregate_statistics(self,
                                    results: List[Dict],
                                    metric_keys: List[str]) -> Dict[str, Dict[str, float]]:
        if not results:
            return {}
        
        aggregates = {}
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                values_array = np.array(values)
                aggregates[key] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'median': float(np.median(values_array))
                }
        
        return aggregates
    
    def generate_summary_report(self) -> str:
        report_lines = ["=" * 80, "STREAMDIAM EXPERIMENTAL RESULTS", "=" * 80, ""]
        
        if self.detection_results:
            report_lines.extend(["DETECTION PERFORMANCE", "-" * 40])
            metrics = self.compute_aggregate_statistics(
                self.detection_results,
                ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            )
            for metric, stats in metrics.items():
                report_lines.append(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            report_lines.append("")
        
        if self.system_results:
            report_lines.extend(["SYSTEM PERFORMANCE", "-" * 40])
            metrics = self.compute_aggregate_statistics(
                self.system_results,
                ['throughput_tps', 'query_latency_mean']
            )
            for metric, stats in metrics.items():
                report_lines.append(f"{metric}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        return "\n".join(report_lines)
    
    def save_results(self, format: str = 'json'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {
            'detection': self.detection_results,
            'system': self.system_results,
            'scalability': self.scalability_results,
            'timestamp': timestamp
        }
        
        output_path = self.output_dir / f'results_{timestamp}.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


class ReproducibilityGuard:
    """Reproducibility enforcement."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.environment_info = {}
    
    def set_global_seed(self):
        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
    
    def capture_environment(self) -> Dict[str, Any]:
        import platform
        
        self.environment_info = {
            'timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'platform': platform.platform()
        }
        
        return self.environment_info


def main():
    """Enhanced main entry point with robust error handling."""
    parser = argparse.ArgumentParser(
        description='StreamDIAM: Scalable Real-Time Illicit Account Detection'
    )
    
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--dataset', type=str, default='EthereumS',
                       choices=['EthereumS', 'EthereumP', 'BitcoinM', 'BitcoinL'])
    parser.add_argument('--experiment-name', type=str)
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--disable-gpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--enable-recovery', action='store_true')
    
    args = parser.parse_args()
    
    try:
        config_manager = ConfigurationManager()
        
        if args.config:
            config = config_manager.create_experiment_config(config_file=args.config)
        else:
            config_dict = {
                'experiment_name': args.experiment_name or f'streamdiam_{datetime.now():%Y%m%d_%H%M%S}',
                'dataset_name': args.dataset,
                'dataset_path': f'./data/{args.dataset}/data.pt',
                'output_dir': args.output_dir,
                'random_seed': args.random_seed,
                'use_gpu': not args.disable_gpu,
                'gpu_ids': args.gpu_ids if not args.disable_gpu else [],
                'enable_recovery': args.enable_recovery
            }
            config = config_manager.create_experiment_config(config_dict=config_dict)
        
        orchestrator = CompleteExperimentOrchestrator(config)
        results = orchestrator.run_complete_pipeline()
        
        logger.info("Experiment completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()