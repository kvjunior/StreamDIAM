# Hardware

The reproduction commands and tests run on commodity CPU hardware; no GPU is
required for the reference implementation or any result in this artifact. A GPU
(CUDA build of torch) only accelerates DIAM fine-tuning and changes no reported
number.

## Platform the committed numbers in this artifact were produced on

- OS: Linux (glibc 2.39), x86-64
- Python: 3.13
- torch 2.15.0 (CPU tensors), NumPy 2.5.3, scikit-learn 1.9.1
- Single process, CPU only

These are a cloud development environment's specs, adequate for the correctness
suite and the scaling-trend runs in `results/`. They are **not** a controlled
benchmarking rig: the absolute latency numbers depend on the machine, so the
paper reports the *ratio* (incremental vs. from-scratch) and the *trend* with
graph size, both of which are machine-robust, rather than leaning on any single
wall-clock value.

## For the camera-ready headline runs

Fill in the exact benchmarking machine here before submission — CPU model and
core count, RAM, GPU model and memory if used, and whether runs were pinned to
dedicated cores — so a reader can match conditions. The full-graph headline runs
(203k-account Elliptic and the DIAM-native graphs) should be produced on that
machine with the native store/maintainer hot path; see the scale note in
`README.md`.
