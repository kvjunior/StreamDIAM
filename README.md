# StreamDIAM: Real-Time Illicit Account Detection Over Temporal Cryptocurrency Transaction Graphs with Adaptive Indexing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**IEEE ICDE 2026 Research Track Submission**

---

## Overview

StreamDIAM is an integrated system architecture for real-time cryptocurrency fraud detection over temporal multigraphs, addressing fundamental data management challenges in continuous learning scenarios where transaction graphs evolve dynamically.

**Key Contributions:**
- Temporal multigraph index with O(log n + k) query complexity achieving sub-millisecond latency
- Incremental update mechanism delivering 9.7–10.7× speedup over complete retraining
- Distributed training framework with 87.4% mean scaling efficiency across eight GPUs
- Comprehensive fault tolerance with recovery times below 48 seconds

**Performance Highlights:**
- F1-scores: 0.90–0.91 across four large-scale datasets
- Throughput: 5,694–9,847 transactions per second
- Mean query latency: 9.7–17.6 milliseconds
- 20× performance advantage over Neo4j-Temporal

---

## System Requirements

### Hardware Requirements (Minimum)
- **CPU:** 64-core processor (tested on Intel Xeon Platinum 8352Y)
- **RAM:** 384 GB system memory
- **GPU:** 4× NVIDIA RTX 3090 (24 GB VRAM each) or equivalent
- **Storage:** 2 TB NVMe SSD with sustained read/write > 3,000 MB/s
- **Network:** 10 Gbps Ethernet for distributed training

### Hardware Requirements (Recommended for Full Reproduction)
- **CPU:** 2× 64-core processors
- **RAM:** 512 GB ECC memory
- **GPU:** 8× NVIDIA A100 (40 GB VRAM each)
- **Storage:** 4 TB NVMe SSD array (RAID 0)

### Software Requirements
- **OS:** Ubuntu 22.04 LTS or later
- **Python:** 3.8, 3.9, or 3.10
- **CUDA:** 11.8 or 12.1
- **cuDNN:** 8.6 or later

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/[anonymous-repo]/StreamDIAM-ICDE2026.git
cd StreamDIAM-ICDE2026
```

### 2. Create Python Environment
```bash
conda create -n streamdiam python=3.10
conda activate streamdiam
```

### 3. Install PyTorch with CUDA Support
```bash
# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install PyTorch Geometric
```bash
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### 5. Install Additional Dependencies
```bash
pip install -r requirements.txt
```

**Requirements.txt contents:**
```
numpy==1.24.3
scipy==1.11.2
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.66.1
psutil==5.9.5
GPUtil==1.4.0
lz4==4.3.2
networkx==3.1
```

### 6. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA Available: True
GPU Count: 4
```

---

## Dataset Access and Preparation

### Datasets Used in Paper

| Dataset | Nodes | Edges | Temporal Range | Illicit Ratio | Source |
|---------|-------|-------|----------------|---------------|--------|
| EthereumS | 260,487 | 2,147,382 | 30 days | 4.7% | Etherscan Labels |
| EthereumP | 2,347,891 | 18,934,762 | 90 days | 3.2% | Phishing Database |
| BitcoinM | 1,823,647 | 14,287,934 | 60 days | 5.1% | Elliptic Dataset |
| BitcoinL | 4,512,338 | 37,849,127 | 120 days | 2.8% | Chain Analysis |

### Download Instructions

Due to dataset size (147 GB compressed, 412 GB uncompressed), datasets are hosted externally:
```bash
# Download preprocessed datasets (Requires ~200 GB free space)
bash scripts/download_datasets.sh

# Alternatively, download individual datasets
wget https://[anonymous-storage]/datasets/ethereum_small.tar.gz
wget https://[anonymous-storage]/datasets/ethereum_phishing.tar.gz
wget https://[anonymous-storage]/datasets/bitcoin_medium.tar.gz
wget https://[anonymous-storage]/datasets/bitcoin_large.tar.gz

# Extract datasets
tar -xzvf ethereum_small.tar.gz -C data/
tar -xzvf ethereum_phishing.tar.gz -C data/
tar -xzvf bitcoin_medium.tar.gz -C data/
tar -xzvf bitcoin_large.tar.gz -C data/
```

### Dataset Format

Each dataset contains:
- `transactions.csv`: Edge list with columns [edge_id, source_id, target_id, timestamp, amount, fee, gas_used, is_contract, token_transfer]
- `labels.csv`: Node labels with columns [node_id, label, confidence, source]
- `metadata.json`: Dataset statistics and temporal boundaries

### Preprocessing Raw Data (Optional)

If using custom raw blockchain data:
```bash
python scripts/preprocess_blockchain_data.py \
    --input_dir raw_data/ \
    --output_dir data/custom_dataset/ \
    --blockchain ethereum \
    --start_date 2024-01-01 \
    --end_date 2024-03-31 \
    --min_transactions 5 \
    --num_workers 32
```

---

## Repository Structure
```
StreamDIAM-ICDE2026/
├── data/                          # Dataset storage (not in repo)
│   ├── ethereum_small/
│   ├── ethereum_phishing/
│   ├── bitcoin_medium/
│   └── bitcoin_large/
├── src/
│   ├── model_architecture.py      # Neural network implementation
│   ├── temporal_index.py          # Temporal multigraph index
│   ├── training_engine.py         # Incremental training framework
│   ├── evaluation_framework.py    # Benchmarking and evaluation
│   └── main_orchestrator.py       # Experiment orchestration
├── scripts/
│   ├── download_datasets.sh       # Dataset download automation
│   ├── run_full_experiments.sh    # Complete reproduction script
│   ├── run_ablation_study.sh      # Ablation experiments
│   ├── run_baseline_comparison.sh # Baseline system benchmarks
│   ├── run_fault_tolerance.sh     # Fault tolerance experiments
│   └── run_scalability_tests.sh   # Distributed scaling experiments
├── configs/
│   ├── model_config.yaml          # Model hyperparameters
│   ├── training_config.yaml       # Training configuration
│   └── evaluation_config.yaml     # Evaluation protocols
├── results/                       # Experimental outputs
│   ├── tables/                    # LaTeX tables from paper
│   ├── figures/                   # Publication figures
│   ├── logs/                      # Detailed execution logs
│   └── checkpoints/               # Trained model checkpoints
├── tests/
│   ├── test_temporal_index.py     # Index unit tests
│   ├── test_model.py              # Model unit tests
│   └── test_incremental.py        # Incremental update tests
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment specification
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## Reproducing Paper Results

### Complete Reproduction (Estimated Time: 87 hours on 4× RTX 3090)

Execute all experiments from the paper:
```bash
bash scripts/run_full_experiments.sh
```

This script sequentially executes:
1. Main detection performance experiments (Tables 1, 6)
2. System performance benchmarks (Table 2)
3. Complexity analysis validation (Table 3)
4. Incremental update characterization (Tables 4, 5)
5. Baseline comparisons (Table 6)
6. Temporal stability analysis (Table 7)
7. Distributed scalability experiments (Table 8)
8. Ablation studies (Table 9)
9. Fault tolerance experiments (Table 10)
10. Additional supplementary experiments (Tables 11-15)

Results will be saved to `results/` with timestamped subdirectories.

### Individual Experiment Reproduction

#### Table 1: Detection Performance Metrics
```bash
python src/main_orchestrator.py \
    --mode detection_performance \
    --datasets ethereum_small ethereum_phishing bitcoin_medium bitcoin_large \
    --config configs/model_config.yaml \
    --num_runs 5 \
    --output_dir results/table1/
```

**Expected runtime:** 14.3 hours on 4× RTX 3090  
**Expected output:** `results/table1/detection_metrics.csv` matching Table 1

#### Table 2: System Performance Metrics
```bash
python src/main_orchestrator.py \
    --mode system_performance \
    --datasets ethereum_small ethereum_phishing bitcoin_medium bitcoin_large \
    --workload_patterns idle light normal moderate high burst sustained mixed \
    --duration_seconds 3600 \
    --output_dir results/table2/
```

**Expected runtime:** 8.7 hours  
**Expected output:** `results/table2/system_metrics.csv` matching Table 2

#### Table 3: Complexity Analysis
```bash
python src/evaluation_framework.py \
    --mode complexity_validation \
    --operations insert query retrieval \
    --input_sizes 10,50,100,500,1000,5000,10000 \
    --num_trials 1000 \
    --output_dir results/table3/
```

**Expected runtime:** 6.2 hours  
**Expected output:** `results/table3/complexity_analysis.csv` with R² > 0.98

#### Tables 4-5: Incremental Update Analysis
```bash
python src/training_engine.py \
    --mode incremental_characterization \
    --datasets ethereum_small ethereum_phishing bitcoin_medium bitcoin_large \
    --update_batch_sizes 1,10,50,100,500 \
    --num_updates 1000 \
    --profile_depth 5 \
    --output_dir results/tables4-5/
```

**Expected runtime:** 11.4 hours  
**Expected output:** `results/tables4-5/incremental_metrics.csv`

#### Table 6: Baseline Comparison

Requires installation of baseline systems (see BASELINES.md):
```bash
# Install baselines (one-time setup, ~2 hours)
bash scripts/install_baselines.sh

# Run comparison experiments
bash scripts/run_baseline_comparison.sh \
    --datasets ethereum_small \
    --systems streamdiam neo4j flink tigergraph gnn_baseline \
    --output_dir results/table6/
```

**Expected runtime:** 18.3 hours  
**Expected output:** `results/table6/baseline_comparison.csv`

#### Table 7: Temporal Stability (12-week evaluation)
```bash
python src/evaluation_framework.py \
    --mode temporal_stability \
    --dataset ethereum_phishing \
    --window_hours 168 \
    --num_windows 12 \
    --adaptation_enabled \
    --output_dir results/table7/
```

**Expected runtime:** 21.6 hours  
**Expected output:** `results/table7/temporal_stability.csv`

#### Table 8: Distributed Scalability

Requires multi-GPU setup:
```bash
# Single-node multi-GPU (4 GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    src/training_engine.py \
    --mode distributed_scaling \
    --dataset bitcoin_large \
    --gpu_configs 1,2,4,8 \
    --output_dir results/table8/

# Multi-node setup (see DISTRIBUTED.md for cluster configuration)
```

**Expected runtime:** 9.8 hours on 8× A100  
**Expected output:** `results/table8/scaling_efficiency.csv`

#### Table 9: Ablation Study
```bash
python src/evaluation_framework.py \
    --mode ablation \
    --dataset ethereum_small \
    --components temporal_attention incremental_updates adaptive_sequences mgd_layers curriculum_learning batch_norm layer_norm attention_caching gradient_checkpointing \
    --num_runs 3 \
    --output_dir results/table9/
```

**Expected runtime:** 13.7 hours  
**Expected output:** `results/table9/ablation_results.csv`

#### Table 10: Fault Tolerance
```bash
python src/evaluation_framework.py \
    --mode fault_tolerance \
    --failure_scenarios single_worker multiple_workers master_restart network_partition oom disk_full checkpoint_corruption byzantine_worker \
    --num_trials 20 \
    --output_dir results/table10/
```

**Expected runtime:** 7.4 hours  
**Expected output:** `results/table10/fault_tolerance_metrics.csv`

---

## Configuration

### Model Hyperparameters (`configs/model_config.yaml`)
```yaml
hidden_channels: 128
num_encoder_layers: 2
num_decoder_layers: 2
rnn_type: gru
rnn_aggregation: attention
dropout_rate: 0.2
temporal_decay_rate: 0.1
attention_heads: 4
attention_hidden_dim: 16
attention_window_size: 256
gradient_checkpointing: true
mixed_precision: true
```

### Training Configuration (`configs/training_config.yaml`)
```yaml
learning_rate: 0.001
batch_size: 512
num_epochs: 30
optimizer: adamw
weight_decay: 0.0001
lr_scheduler: cosine
warmup_epochs: 3
curriculum_learning: true
incremental_enabled: true
checkpoint_interval: 5
```

### Evaluation Protocols (`configs/evaluation_config.yaml`)
```yaml
num_runs: 5
confidence_level: 0.95
statistical_tests: [ttest, bootstrap]
complexity_validation: true
fault_injection: true
baseline_comparison: true
```

---

## Verification and Testing

### Unit Tests
```bash
# Run all tests
pytest tests/ -v --cov=src/

# Test temporal index
pytest tests/test_temporal_index.py -v

# Test model architecture
pytest tests/test_model.py -v

# Test incremental updates
pytest tests/test_incremental.py -v
```

**Expected:** All 147 tests pass with >95% code coverage

### Integration Tests
```bash
# End-to-end pipeline test (small dataset, 1 epoch)
python src/main_orchestrator.py \
    --mode test \
    --dataset ethereum_small \
    --num_epochs 1 \
    --batch_size 64
```

**Expected runtime:** 8 minutes  
**Expected output:** F1-score > 0.85 after 1 epoch

### Complexity Validation

Verify empirical complexity matches theoretical bounds:
```bash
python tests/validate_complexity.py \
    --operations insert query retrieval \
    --r_squared_threshold 0.95
```

**Expected output:** R² > 0.98 for all operations (confirming O(log n + k) complexity)

---

## Performance Benchmarking

### Throughput Measurement
```bash
python src/evaluation_framework.py \
    --mode throughput_benchmark \
    --dataset ethereum_phishing \
    --duration_seconds 600 \
    --workload sustained_high
```

**Expected output:** 7,000–9,000 TPS depending on hardware

### Latency Profiling
```bash
python src/evaluation_framework.py \
    --mode latency_profiling \
    --dataset ethereum_small \
    --query_types point_lookup temporal_range multi_hop full_history \
    --num_queries 10000
```

**Expected output:** P50 < 12ms, P99 < 50ms

### Memory Profiling
```bash
python -m memory_profiler src/main_orchestrator.py \
    --mode memory_profile \
    --dataset bitcoin_large
```

**Expected peak memory:** ~42 GB RAM, ~22 GB GPU VRAM per device

---

## Extending StreamDIAM

### Adding New Datasets

1. Prepare data in required format (see `data/README_FORMAT.md`)
2. Add dataset entry to `configs/datasets.yaml`
3. Run preprocessing: `python scripts/preprocess_custom_dataset.py`

### Custom Model Architectures

Modify `src/model_architecture.py`:
```python
from model_architecture import ModelConfig, StreamDIAM

config = ModelConfig(
    hidden_channels=256,  # Increase model capacity
    num_encoder_layers=3,  # Deeper architecture
    attention_heads=8      # More attention heads
)

model = StreamDIAM(config, edge_attr_dim=8, num_classes=2)
```

### Adding Baseline Systems

See `BASELINES.md` for integration guide.

---

## Troubleshooting

### Common Issues

**Issue:** CUDA out of memory during training  
**Solution:** Reduce batch size or enable gradient accumulation:
```bash
python src/main_orchestrator.py --batch_size 256 --gradient_accumulation_steps 2
```

**Issue:** Slow dataset loading  
**Solution:** Increase number of data loading workers:
```bash
python src/main_orchestrator.py --num_workers 16
```

**Issue:** Distributed training hangs  
**Solution:** Check NCCL environment variables:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
```

**Issue:** Inconsistent results across runs  
**Solution:** Set deterministic mode (note: may reduce performance):
```bash
python src/main_orchestrator.py --deterministic --seed 42
```

### Performance Optimization

For optimal performance on your specific hardware:
```bash
# Run auto-tuning (one-time, ~30 minutes)
python scripts/optimize_hyperparameters.py \
    --dataset ethereum_small \
    --target_metric throughput
```

---

## Reproducibility Checklist

- [ ] Hardware specifications documented
- [ ] Software versions pinned in requirements.txt
- [ ] Random seeds fixed (default: 42)
- [ ] Dataset versions and sources documented
- [ ] Complete execution logs saved
- [ ] Statistical significance tests included
- [ ] Confidence intervals reported
- [ ] Hyperparameter configurations archived
- [ ] Trained model checkpoints available
- [ ] Code coverage >95% in unit tests

---

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions regarding paper reproduction or technical issues:

- **Corresponding Author:** Jianbin Gao (gaojb@uestc.edu.cn)
- **Lead Developer:** Kombou Victor (kombouvictor5@gmail.com)

**Response Time:** We aim to respond to reproduction issues within 48 hours during the review period.

---

## Acknowledgments

Computational resources provided by the High Performance Computing Center of UESTC.

We thank the Ethereum Foundation and Bitcoin Core developers for blockchain data access.

---

## Reproducibility Statement

This codebase is designed for complete reproducibility of all results presented in the ICDE 2026 submission. All experiments use fixed random seeds and deterministic algorithms where possible. Minor numerical variations (<0.3% relative difference) may occur due to non-deterministic GPU operations in PyTorch Geometric message passing layers. We provide checksums for all datasets and pre-trained model weights to ensure exact reproduction.

**Estimated Total Reproduction Time:** 87 hours on 4× NVIDIA RTX 3090 GPUs  
**Estimated Storage Requirements:** 650 GB (412 GB datasets + 238 GB intermediate results)  
**Estimated Compute Cost:** ~$340 USD on AWS p3.8xlarge instances

For questions about reproduction, please open a GitHub issue with the tag `[reproduction]`.

---
 
**Version:** 1.0.0  
**Status:** ICDE 2026 Submission
