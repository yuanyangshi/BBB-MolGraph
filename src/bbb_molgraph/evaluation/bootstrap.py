"""
Bootstrap statistical estimation providing 95% confidence intervals for publication figures.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from bbb_molgraph.evaluation.metrics import compute_classification_metrics


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str = "roc_auc",
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Calculate empirical Bootstrap mean, standard deviation, and confidence interval.

    Args:
        y_true: True binary labels (N,)
        y_prob: Predicted probabilities (N,)
        metric_name: Key in compute_classification_metrics
        n_bootstraps: Number of bootstrap iterations (recommended: 1000)
        ci: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed for sampling reproducibility

    Returns:
        Dictionary containing 'mean', 'std', 'ci_lower', 'ci_upper', and 'formatted'.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n_samples = len(y_true)

    rng = np.random.RandomState(seed)
    bootstrapped_scores: List[float] = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        # Ensure sample contains both classes
        if len(np.unique(y_true[indices])) < 2:
            continue

        metrics = compute_classification_metrics(y_true[indices], y_prob[indices])
        score = metrics.get(metric_name, float("nan"))
        if not np.isnan(score):
            bootstrapped_scores.append(score)

    if not bootstrapped_scores:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "formatted": "N/A",
        }

    scores_arr = np.array(bootstrapped_scores)
    mean_val = float(np.mean(scores_arr))
    std_val = float(np.std(scores_arr))

    alpha = (1.0 - ci) / 2.0
    lower_bound = float(np.percentile(scores_arr, alpha * 100))
    upper_bound = float(np.percentile(scores_arr, (1.0 - alpha) * 100))

    formatted_str = f"{mean_val:.3f} ± {std_val:.3f} [{lower_bound:.3f}, {upper_bound:.3f}]"

    return {
        "mean": mean_val,
        "std": std_val,
        "ci_lower": lower_bound,
        "ci_upper": upper_bound,
        "formatted": formatted_str,
    }


def compute_all_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute 95% Bootstrap Confidence Intervals for all primary metrics.
    """
    primary_metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "mcc", "sensitivity", "specificity", "f1"]
    results = {}
    for metric in primary_metrics:
        results[metric] = bootstrap_confidence_interval(
            y_true=y_true,
            y_prob=y_prob,
            metric_name=metric,
            n_bootstraps=n_bootstraps,
            ci=ci,
            seed=seed,
        )
    return results
