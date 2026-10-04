"""
Training callbacks: Early Stopping, Checkpointing, and Metric Tracking.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from bbb_molgraph.utils.logging import logger


class Callback:
    """Base callback interface."""

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> bool:
        """Return True to stop training early."""
        return False

    def on_train_end(self, trainer: Any) -> None:
        pass


class EarlyStopping(Callback):
    """
    Early stopping to halt training when monitored metric stops improving.
    """

    def __init__(
        self,
        monitor: str = "val_roc_auc",
        mode: str = "max",
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> bool:
        current = metrics.get(self.monitor)
        if current is None:
            return False

        if self.best_score is None:
            self.best_score = current
            return False

        if self.mode == "max":
            improved = current > (self.best_score + self.min_delta)
        else:
            improved = current < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}: '{self.monitor}' "
                    f"did not improve for {self.patience} consecutive epochs."
                )
                self.should_stop = True
                return True

        return False


class ModelCheckpoint(Callback):
    """
    Saves model checkpoints whenever the monitored metric improves.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = "val_roc_auc",
        mode: str = "max",
        save_best_only: bool = True,
    ) -> None:
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score: Optional[float] = None

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> bool:
        current = metrics.get(self.monitor)
        if current is None:
            return False

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = current > self.best_score
        else:
            improved = current < self.best_score

        if improved:
            self.best_score = current
            save_payload = {
                "epoch": epoch,
                "state_dict": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "metrics": metrics,
            }
            torch.save(save_payload, self.filepath)
            logger.info(f"Checkpoint saved to {self.filepath} ({self.monitor}: {current:.4f})")

        return False
