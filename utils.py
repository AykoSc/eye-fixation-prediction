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
