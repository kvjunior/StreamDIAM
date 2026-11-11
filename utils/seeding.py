"""Global seed control for reproducible runs.

Reproducibility is part of the evaluation (R1 O18): a fine-tune or an experiment
must be re-runnable to the same numbers. :func:`seed_everything` seeds every RNG
the pipeline touches -- Python, NumPy, and Torch (CPU + CUDA) -- and optionally
puts Torch in deterministic mode.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and Torch RNGs; return the seed for logging.

    Args:
        seed: the seed applied to every RNG.
        deterministic: if set, also request deterministic cuDNN/Torch algorithms
            (slower, but bit-reproducible where the ops support it).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
