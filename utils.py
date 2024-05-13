import torch
import torch_directml as tdm
from logger import logger


def get_device():
    if tdm.is_available():
        logger.info("DirectML device is available.")
        return tdm.device()
    elif torch.cuda.is_available():
        logger.info("CUDA device is available.")
        return torch.device("cuda")
    else:
        logger.info("Neither DirectML nor CUDA is available. Using CPU.")
        return torch.device("cpu")


def gaussian(window_size: int, sigma: float, device=None) -> torch.Tensor:
    dtype = None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    device = device or torch.device('cpu')
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()
