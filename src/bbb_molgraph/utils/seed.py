"""
Deterministic random seed configuration ensuring absolute scientific reproducibility.
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_deterministic_seed(seed: int = 42, deterministic_algorithms: bool = True) -> None:
    """
    Enforce strict reproducibility across Python, NumPy, PyTorch, and CUDA.

    Args:
        seed: Integer random seed (default: 42).
        deterministic_algorithms: Whether to enable PyTorch deterministic algorithms flag.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Required for deterministic execution on CUDA >= 10.2
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            # Graceful fallback on hardware/operations that do not implement deterministic kernels
            pass


def get_worker_init_fn(seed: int = 42):
    """
    Returns a worker initialization function for PyTorch DataLoader to ensure deterministic workers.
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn
