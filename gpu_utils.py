"""
GPU utilities for detecting and configuring CUDA acceleration.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Tuple[bool, str]:
    """
    Check if GPU is available for CUDA acceleration.
    
    Returns:
        Tuple of (is_available, device_info)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            info = f"{device_name} ({device_count} GPU(s), {memory_gb:.1f} GB)"
            return True, info
        else:
            return False, "CUDA not available"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, str(e)


def get_gpu_memory() -> float:
    """Get available GPU memory in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return 0.0
    except:
        return 0.0


def get_optimal_gpu_config(gpu_memory_gb: float) -> dict:
    """
    Get optimal GPU configuration based on available memory.
    
    Args:
        gpu_memory_gb: Total GPU memory in GB
    
    Returns:
        Dictionary of recommended settings
    """
    config = {
        "use_gpu": gpu_memory_gb >= 4,
        "num_rays_per_batch": 4096,  # Default
        "use_mixed_precision": gpu_memory_gb >= 6,
    }
    
    # Optimize based on available memory
    if gpu_memory_gb >= 24:
        config["num_rays_per_batch"] = 16384
        config["use_mixed_precision"] = True
    elif gpu_memory_gb >= 12:
        config["num_rays_per_batch"] = 8192
        config["use_mixed_precision"] = True
    elif gpu_memory_gb >= 8:
        config["num_rays_per_batch"] = 6144
        config["use_mixed_precision"] = True
    elif gpu_memory_gb >= 6:
        config["num_rays_per_batch"] = 4096
        config["use_mixed_precision"] = True
    elif gpu_memory_gb >= 4:
        config["num_rays_per_batch"] = 2048
        config["use_mixed_precision"] = False
    
    return config


def get_gpu_nerfstudio_args(use_gpu: bool = True, mixed_precision: bool = False) -> list:
    """
    Generate Nerfstudio command line arguments for GPU acceleration.
    
    Note: Nerfstudio auto-detects GPU. This function returns minimal compatible args.
    
    Args:
        use_gpu: Whether to use GPU (nerfstudio auto-detects)
        mixed_precision: Whether to use mixed precision (not all versions support this)
    
    Returns:
        List of command line arguments compatible with the installed nerfstudio
    """
    args = []
    
    # Nerfstudio automatically detects and uses GPU if available
    # No need to specify --machine.num-gpus since it auto-detects CUDA
    
    # Note: Mixed precision and other advanced options may not be available
    # in all nerfstudio versions, so we only add them if explicitly supported
    
    return args
