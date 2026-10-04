"""
Learning rate scheduling utilities including linear warmup with cosine decay.
"""

import math
from typing import Any, Dict
import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, StepLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_warmup",
    **kwargs: Any,
) -> Any:
    """
    Factory function for learning rate schedulers.
    """
    if scheduler_type == "cosine_warmup":
        warmup_steps = kwargs.get("warmup_steps", 5)
        total_steps = kwargs.get("total_steps", 50)
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "max"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
        )
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.5),
        )
    return None
