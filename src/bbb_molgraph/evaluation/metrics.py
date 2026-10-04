"""
Rigorous evaluation metrics conforming to top-tier journal standards in chemoinformatics and drug discovery.
"""

from typing import Dict, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor, list],
    y_prob: Union[np.ndarray, torch.Tensor, list],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for binary BBB permeability prediction.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (BBB+).
        threshold: Classification threshold for binarizing probabilities (default: 0.5).

    Returns:
        Dictionary containing:
        - roc_auc: Area Under the Receiver Operating Characteristic Curve
        - pr_auc: Area Under the Precision-Recall Curve (Average Precision)
        - balanced_accuracy: Balanced Accuracy Score
        - mcc: Matthews Correlation Coefficient
        - sensitivity: Recall of positive class (BBB+)
        - specificity: Recall of negative class (BBB-)
        - f1: F1 harmonic mean score
        - precision: Positive predictive value
        - accuracy: Standard overall accuracy
        - brier_score: Calibration Brier Score (lower is better)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Sensitivity (True Positive Rate) & Specificity (True Negative Rate)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")

    mcc = float(matthews_corrcoef(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "precision": precision,
        "accuracy": acc,
        "brier_score": brier,
    }
