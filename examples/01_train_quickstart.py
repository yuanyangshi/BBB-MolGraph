"""
Example 01: Quickstart training demonstration of BBB-MolGraph.
"""

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bbb_molgraph.data.dataset import MolecularGraphDataset, collate_molecular_graphs
from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer
from bbb_molgraph.models.mpnn import MPNNClassifier
from bbb_molgraph.training.callbacks import EarlyStopping, ModelCheckpoint
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
    if not data_csv.exists():
        data_csv = root / "data" / "sample_molecules.csv"

    logger.info(f"Loading data from {data_csv}")
    df = pd.read_csv(data_csv)

    # Use a small subset for fast demonstration
    df_subset = df.sample(n=200, random_state=42).reset_index(drop=True)
    smiles_list = df_subset["smiles"].tolist()
    labels = df_subset["label"].values

    featurizer = MolecularGraphFeaturizer()
    split_idx = int(len(smiles_list) * 0.8)

    train_dataset = MolecularGraphDataset(
        smiles_list=smiles_list[:split_idx],
        labels=labels[:split_idx],
        featurizer=featurizer,
    )
    val_dataset = MolecularGraphDataset(
        smiles_list=smiles_list[split_idx:],
        labels=labels[split_idx:],
        featurizer=featurizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_molecular_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_molecular_graphs,
    )

    model = MPNNClassifier(
        atomic_dim=featurizer.atom_dim,
        bond_dim=featurizer.bond_dim,
        hidden_dim=32,
        propagation_steps=3,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    checkpoint_path = root / "checkpoints" / "quickstart_model.pt"
    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, monitor="roc_auc", mode="max"),
        EarlyStopping(monitor="roc_auc", patience=5),
    ]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=callbacks,
    )

    logger.info("Training MPNN for 3 epochs...")
    history = trainer.fit(train_loader, val_loader, max_epochs=3)
    logger.info("Quickstart training completed successfully.")


if __name__ == "__main__":
    main()
