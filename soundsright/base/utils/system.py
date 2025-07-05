import shutil
import os
import torch

def get_gpu_count():
    """Returns the number of GPUs available using PyTorch."""
    return torch.cuda.device_count()

def get_cpu_core_count():
    """Returns the number of CPU cores available on the system."""
    return os.cpu_count()

def get_free_space_gb(path='/'):
    """Returns the available free disk space at the given path in GB."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    return round(free_gb, 2)

