import logging

import torch
from torch import nn


def load_checkpoint(model: nn.Module, filepath: str, *, strict: bool = True) -> nn.Module:
    """Load model checkpoint file.

    Args:
        model (nn.Module): Module to load checkpoint.
        filepath (str): File path of checkpoint.
        strict (bool, optional): Whether to strictly enforce that keys in `state_dict` match keys in checkpoint.

    Returns:
        nn.Module: Module after loading the checkpoint.
    """
    checkpoint: dict = torch.load(filepath, map_location=torch.device("cpu"), weights_only=True)[
        "model"
    ]

    state_dict: dict = model.module.state_dict() if is_parallel(model) else model.state_dict()

    load_state = {}
    for name, weight in state_dict.items():
        weight: torch.Tensor
        if name not in checkpoint:
            logging.warning(f"{name} is not contained in the checkpoint.")
            continue
        ckpt_weight: torch.Tensor = checkpoint[name]
        if weight.shape != ckpt_weight.shape:
            logging.warning(
                f"Shape of {name} in checkpoint {ckpt_weight.shape}, while shape of {name} in model is {weight.shape}"
            )
            continue
        load_state[name] = ckpt_weight

    if is_parallel(model):
        model.module.load_state_dict(load_state, strict=strict)
    else:
        model.load_state_dict(load_state, strict=strict)

    return model


def is_parallel(model: nn.Module) -> bool:
    """Check whether the model is in parallel.

    Args:
        model (nn.Module): _des

    Returns:
        bool: _description_
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))
