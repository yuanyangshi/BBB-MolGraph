"""
Scientific Trainer for molecular deep learning models with AMP and early stopping.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bbb_molgraph.evaluation.metrics import compute_classification_metrics
from bbb_molgraph.training.callbacks import Callback
from bbb_molgraph.utils.logging import logger


class Trainer:
    """
    Standardized trainer supporting AMP mixed-precision, gradient clipping,
    and automatic validation with full chemoinformatics metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callback]] = None,
        lr_scheduler: Optional[Any] = None,
        use_amp: bool = True,
        grad_clip_norm: float = 5.0,
    ) -> None:
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks or []
        self.lr_scheduler = lr_scheduler
        self.grad_clip_norm = grad_clip_norm

        # Enable AMP only when CUDA is available
        self.use_amp = use_amp and (self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _forward_model(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Dynamically route batch to model based on input modality.
        """
        batch_graph = {
            "node_features": batch["node_features"].to(self.device),
            "edge_features": batch["edge_features"].to(self.device),
            "connectivity_indices": batch["connectivity_indices"].to(self.device),
            "batch_assignment": batch["batch_assignment"].to(self.device),
        }

        # Dual-channel model
        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = (
                batch["attention_mask"].to(self.device)
                if "attention_mask" in batch
                else None
            )
            out = self.model(batch_graph, input_ids, attention_mask)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            # Single-channel graph model
            out = self.model(batch_graph)
            logits = out[0] if isinstance(out, tuple) else out

        return logits

    def _autocast_context(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            targets = batch["labels"].to(self.device)
            self.optimizer.zero_grad()

            with self._autocast_context():
                logits = self._forward_model(batch)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run evaluation on validation or test set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_targets: List[float] = []
        all_probs: List[float] = []

        for batch in val_loader:
            targets = batch["labels"].to(self.device)
            with self._autocast_context():
                logits = self._forward_model(batch)
                loss = self.criterion(logits, targets)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits).cpu().view(-1).tolist()
            labels = targets.cpu().view(-1).tolist()
            all_probs.extend(probs)
            all_targets.extend(labels)

        metrics = compute_classification_metrics(all_targets, all_probs)
        metrics["val_loss"] = total_loss / max(1, num_batches)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 50,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fit model across epochs, executing callbacks and recording history.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_pr_auc": [],
            "val_mcc": [],
        }

        for cb in self.callbacks:
            cb.on_train_begin(self)

        iterator = range(1, max_epochs + 1)
        if verbose:
            iterator = tqdm(iterator, desc="Training Epochs")

        for epoch in iterator:
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics.get("val_loss", 0.0))
            history["val_roc_auc"].append(val_metrics.get("roc_auc", 0.0))
            history["val_pr_auc"].append(val_metrics.get("pr_auc", 0.0))
            history["val_mcc"].append(val_metrics.get("mcc", 0.0))

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metrics.get("roc_auc", 0.0))
                else:
                    self.lr_scheduler.step()

            if verbose:
                logger.info(
                    f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                    f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                    f"Val MCC: {val_metrics['mcc']:.4f}"
                )

            stop_flags = [cb.on_epoch_end(self, epoch, val_metrics) for cb in self.callbacks]
            if any(stop_flags):
                logger.info(f"Stopping early at epoch {epoch}")
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

        return history
