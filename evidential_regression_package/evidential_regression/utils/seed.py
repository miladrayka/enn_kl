import os
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and Python.

    Parameters
    ----------
    seed : int, optional
        Seed value to set for random number generators, by default 42.

    Notes
    -----
    This function ensures deterministic behavior for CPU and GPU computations
    in PyTorch and sets the PYTHONHASHSEED environment variable.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
