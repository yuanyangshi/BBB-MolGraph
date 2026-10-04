"""
Unit tests for Trainer, early stopping, metrics, and bootstrap statistics.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from bbb_molgraph.data.dataset import MolecularGraphDataset, collate_molecular_graphs
from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer
from bbb_molgraph.evaluation.bootstrap import bootstrap_confidence_interval
from bbb_molgraph.evaluation.metrics import compute_classification_metrics
from bbb_molgraph.models.mpnn import MPNNClassifier
from bbb_molgraph.training.callbacks import EarlyStopping
from bbb_molgraph.training.trainer import Trainer


def test_metrics_calculation():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
    metrics = compute_classification_metrics(y_true, y_prob)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_bootstrap_ci():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15, 0.6, 0.4])
    ci_res = bootstrap_confidence_interval(y_true, y_prob, metric_name="roc_auc", n_bootstraps=50)

    assert "mean" in ci_res
    assert "ci_lower" in ci_res
    assert "ci_upper" in ci_res
    assert ci_res["ci_lower"] <= ci_res["mean"] <= ci_res["ci_upper"]


def test_trainer_single_epoch_convergence():
    featurizer = MolecularGraphFeaturizer()
    smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN"]
    labels = [1.0, 1.0, 0.0, 0.0]

    ds = MolecularGraphDataset(smiles, labels, featurizer=featurizer)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_molecular_graphs)

    model = MPNNClassifier(
        atomic_dim=featurizer.atom_dim,
        bond_dim=featurizer.bond_dim,
        hidden_dim=16,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion)
    loss1 = trainer.train_epoch(loader)
    loss2 = trainer.train_epoch(loader)

    # Overfitting on tiny 4-item dataset should decrease loss
    assert isinstance(loss1, float)
    assert isinstance(loss2, float)


def test_early_stopping_callback():
    es = EarlyStopping(monitor="val_roc_auc", mode="max", patience=2)
    # Simulate decreasing scores
    stop1 = es.on_epoch_end(None, 1, {"val_roc_auc": 0.8})
    assert not stop1
    stop2 = es.on_epoch_end(None, 2, {"val_roc_auc": 0.75})
    assert not stop2
    stop3 = es.on_epoch_end(None, 3, {"val_roc_auc": 0.70})
    assert stop3  # Triggered after 2 stagnant/declining epochs
