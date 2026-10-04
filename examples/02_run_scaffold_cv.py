"""
Example 02: 5-Fold Stratified Bemis-Murcko Scaffold Cross-Validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bbb_molgraph.data.dataset import MolecularGraphDataset, collate_molecular_graphs
from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer
from bbb_molgraph.data.splitter import ScaffoldSplitter
from bbb_molgraph.evaluation.bootstrap import compute_all_bootstrap_cis
from bbb_molgraph.models.mpnn import MPNNClassifier
from bbb_molgraph.training.trainer import Trainer
from bbb_molgraph.utils.config_parser import get_project_root
from bbb_molgraph.utils.logging import logger
from bbb_molgraph.utils.seed import set_deterministic_seed


def main() -> None:
    set_deterministic_seed(42)
    root = get_project_root()
    data_csv = root / "data" / "raw" / "BBBP_combined.csv"
    if not data_csv.exists():
        data_csv = root / "data" / "BBBP_combined.csv"
    if not data_csv.exists():
        data_csv = root.parent / "data" / "BBBP_combined.csv"

    df = pd.read_csv(data_csv)
    smiles_list = df["smiles"].tolist()
    labels = df["label"].values

    splits_file = root / "data" / "splits" / "scaffold_folds.json"
    if splits_file.exists():
        folds = ScaffoldSplitter.load_folds_from_json(splits_file)
    else:
        splitter = ScaffoldSplitter()
        folds = splitter.k_fold_scaffold_cv(smiles_list, k=5, seed=42)
        splits_file.parent.mkdir(parents=True, exist_ok=True)
        ScaffoldSplitter.save_folds_to_json(folds, splits_file)

    logger.info(f"Loaded {len(folds)} scaffold splits from {splits_file}")

    featurizer = MolecularGraphFeaturizer()
    fold_metrics = []

    # Run fold 0 as demonstration (can be extended to all folds)
    for fold_id, (train_idx, test_idx) in enumerate(folds[:1]):
        logger.info(f"--- Running Fold {fold_id} ---")
        train_ds = MolecularGraphDataset(
            smiles_list=[smiles_list[i] for i in train_idx[:300]],
            labels=[labels[i] for i in train_idx[:300]],
            featurizer=featurizer,
        )
        test_ds = MolecularGraphDataset(
            smiles_list=[smiles_list[i] for i in test_idx[:100]],
            labels=[labels[i] for i in test_idx[:100]],
            featurizer=featurizer,
        )

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_molecular_graphs)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_molecular_graphs)

        model = MPNNClassifier(atomic_dim=featurizer.atom_dim, bond_dim=featurizer.bond_dim, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(model=model, optimizer=optimizer, criterion=torch.nn.BCEWithLogitsLoss())

        trainer.fit(train_loader, test_loader, max_epochs=3, verbose=False)
        metrics = trainer.evaluate(test_loader)
        logger.info(f"Fold {fold_id} Test ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
        fold_metrics.append(metrics)

    logger.info("Scaffold cross-validation demonstration finished.")


if __name__ == "__main__":
    main()
