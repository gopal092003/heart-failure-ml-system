import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    # For hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_deterministic(seed: int = 42) -> None:
    """
    Enforce deterministic behavior across libraries.
    """
    set_seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # Torch not used


def log_seed(seed: int, logger=None):
    """
    Log seed for reproducibility tracking.
    """
    if logger:
        logger.info(f"Using random seed: {seed}")
    else:
        print(f"[INFO] Using random seed: {seed}")