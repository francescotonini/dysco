import logging
import os
import random
import warnings

import numpy as np
import torch

from src.utils import logger

log = logger.get_logger(__name__)


def set_seed(seed: int) -> None:

    # Check seed is valid
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")
    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    log.info(f"Setting seed to {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    # torch.use_deterministic_algorithms(True)


def init(cfg):
    # Disable httpx logging
    logging.getLogger("httpx").propagate = False

    # Setup seed
    if cfg.get("seed"):
        set_seed(cfg.seed)

    # Disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # Torch float32 precision
    if cfg.get("float32_matmul_precision"):
        log.info(f"Setting torch.float32_matmul_precision to {cfg.float32_matmul_precision}")
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # Pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        logger.print_config_tree(cfg, resolve=True)

    # Create logs directory
    os.makedirs(os.path.join(cfg.paths.output_dir, "logs"), exist_ok=True)


def default_collate_fn(batch: list[dict]) -> dict:
    assert isinstance(batch[0], dict)

    collated = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]

    return collated


def iter_chunks(items, items_per_chunk):
    for i in range(0, len(items), items_per_chunk):
        yield items[i : i + items_per_chunk]
