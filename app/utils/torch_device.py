import torch
from loguru import logger

__all__ = ["get_torch_device"]


def get_torch_device() -> torch.device:
    """Provide best possible device for running PyTorch."""
    if torch.cuda.is_available():
        gpu0 = torch.cuda.get_device_name(0)
        logger.info(f"CUDA is available. Running PyTorch on GPU ({gpu0}).")
        return torch.device("cuda")
    else:
        logger.info(f"Running PyTorch on CPU.")
        return torch.device("cpu")
