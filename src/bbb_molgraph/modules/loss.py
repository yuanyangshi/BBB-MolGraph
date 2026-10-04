"""
Loss functions customized for imbalanced binary molecular classification.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss focusing learning on hard negative/positive examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy with automatic or custom positive weight compensation.
    """

    def __init__(self, pos_weight: Optional[float] = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.pos_weight_tensor: Optional[torch.Tensor] = torch.tensor([pos_weight])
        else:
            self.pos_weight_tensor = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.pos_weight_tensor.to(logits.device) if self.pos_weight_tensor is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
