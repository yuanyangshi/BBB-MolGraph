"""
Command-Line Interface (CLI) for training, evaluation, and batch prediction.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bbb_molgraph.data.dataset import DualChannelCollate, MolecularGraphDataset, collate_molecular_graphs
from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer
from bbb_molgraph.data.splitter import ScaffoldSplitter
from bbb_molgraph.evaluation.bootstrap import compute_all_bootstrap_cis
from bbb_molgraph.evaluation.metrics import compute_classification_metrics
from bbb_molgraph.models import DualChannelBBBPredictor, MPNNClassifier, load_model_from_checkpoint
from bbb_molgraph.training.callbacks import EarlyStopping, ModelCheckpoint
from bbb_molgraph.training.trainer import Trainer
from bbb_molgraph.utils.config_parser import get_project_root, load_yaml_config, parse_cli_overrides
from bbb_molgraph.utils.logging import logger
from bbb_molgraph.utils.seed import get_worker_init_fn, set_deterministic_seed


def train_cli(args: Optional[list] = None) -> None:
    """Entry point for bbb-train command."""
    parser = argparse.ArgumentParser(description="Train BBB-MolGraph models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/scaffold_5fold_cv.yaml",
        help="Path to YAML experiment configuration file.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in key=value format (e.g. training.lr=0.0005).",
    )
    parsed_args = parser.parse_args(args)

    config = load_yaml_config(Path(parsed_args.config))
    if parsed_args.overrides:
        from bbb_molgraph.utils.config_parser import deep_merge
        overrides = parse_cli_overrides(parsed_args.overrides)
        config = deep_merge(config, overrides)

    seed = config.get("seed", 42)
    set_deterministic_seed(seed)

    data_cfg = config.get("data", {})
    data_path = Path(data_cfg.get("csv_path", "data/raw/BBBP_combined.csv"))
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path

    if not data_path.exists():
        alt_path = get_project_root() / "data" / "BBBP_combined.csv"
        workspace_path = get_project_root().parent / "data" / "BBBP_combined.csv"
        if alt_path.exists():
            data_path = alt_path
        elif workspace_path.exists():
            data_path = workspace_path
        else:
            raise FileNotFoundError(
                f"Data file not found at {data_path} or {workspace_path}. "
                "Please download the dataset per instructions in data/README.md."
            )

    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    smiles_col = data_cfg.get("smiles_column", "smiles")
    target_col = data_cfg.get("target_column", "label")

    smiles_list = df[smiles_col].tolist()
    labels = df[target_col].values

    splitter = ScaffoldSplitter()
    splits_file = get_project_root() / "data" / "splits" / "scaffold_folds.json"

    if splits_file.exists():
        logger.info(f"Loading precomputed scaffold splits from {splits_file}")
        folds = ScaffoldSplitter.load_folds_from_json(splits_file)
    else:
        logger.info("Computing 5-fold Bemis-Murcko scaffold splits...")
        folds = splitter.k_fold_scaffold_cv(smiles_list, k=5, seed=seed)
        ScaffoldSplitter.save_folds_to_json(folds, splits_file)

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    max_epochs = train_cfg.get("max_epochs", 20)
    lr = float(train_cfg.get("lr", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))

    featurizer = MolecularGraphFeaturizer()
    model_type = config.get("model", {}).get("type", "mpnn")

    logger.info(f"Initializing {model_type} model architecture...")
    if model_type == "dual_channel":
        model = DualChannelBBBPredictor(
            atomic_dim=featurizer.atom_dim,
            bond_dim=featurizer.bond_dim,
            use_fallback_molformer=True,  # Default to fallback in CLI for instant offline run
        )
        collate_fn = DualChannelCollate(tokenizer=model.seq_encoder.tokenizer)
    else:
        model = MPNNClassifier(
            atomic_dim=featurizer.atom_dim,
            bond_dim=featurizer.bond_dim,
        )
        collate_fn = collate_molecular_graphs

    # Train on Fold 0 as demonstration or full CV if specified
    fold_0_train_idx, fold_0_val_idx = folds[0]
    train_dataset = MolecularGraphDataset(
        smiles_list=[smiles_list[i] for i in fold_0_train_idx],
        labels=[labels[i] for i in fold_0_train_idx],
        featurizer=featurizer,
    )
    val_dataset = MolecularGraphDataset(
        smiles_list=[smiles_list[i] for i in fold_0_val_idx],
        labels=[labels[i] for i in fold_0_val_idx],
        featurizer=featurizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=get_worker_init_fn(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    checkpoint_dir = get_project_root() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoint_dir / f"{model_type}_best.pt"

    callbacks = [
        ModelCheckpoint(filepath=best_ckpt, monitor="roc_auc", mode="max"),
        EarlyStopping(monitor="roc_auc", mode="max", patience=10),
    ]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=callbacks,
    )

    logger.info("Starting model training...")
    history = trainer.fit(train_loader, val_loader, max_epochs=max_epochs)
    logger.info(f"Training completed. Best model checkpoint saved to: {best_ckpt}")


def eval_cli(args: Optional[list] = None) -> None:
    """Entry point for bbb-eval command."""
    parser = argparse.ArgumentParser(description="Evaluate BBB-MolGraph checkpoint.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint .pt file.")
    parser.add_argument(
        "--data-csv",
        type=str,
        default="data/raw/BBBP_combined.csv",
        help="Path to evaluation CSV.",
    )
    parsed_args = parser.parse_args(args)

    data_path = Path(parsed_args.data_csv)
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path

    df = pd.read_csv(data_path)
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[3]
    label_col = "label" if "label" in df.columns else df.columns[2]

    featurizer = MolecularGraphFeaturizer()
    dataset = MolecularGraphDataset(
        smiles_list=df[smiles_col].tolist(),
        labels=df[label_col].values,
        featurizer=featurizer,
    )
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_molecular_graphs)

    model = load_model_from_checkpoint(parsed_args.model_path)
    if isinstance(model, DualChannelBBBPredictor):
        collate_fn = DualChannelCollate(tokenizer=model.seq_encoder.tokenizer)
    else:
        collate_fn = collate_molecular_graphs

    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
    trainer = Trainer(model=model, optimizer=torch.optim.Adam(model.parameters()), criterion=torch.nn.BCEWithLogitsLoss())
    metrics = trainer.evaluate(loader)

    logger.info("=== Evaluation Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


def predict_cli(args: Optional[list] = None) -> None:
    """Entry point for bbb-predict command."""
    parser = argparse.ArgumentParser(description="Predict BBB permeability for molecules.")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to input CSV containing SMILES.")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to save predictions CSV.")
    parser.add_argument("--smiles-col", type=str, default="smiles", help="Column name for SMILES.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional model checkpoint path.")
    parsed_args = parser.parse_args(args)

    df = pd.read_csv(parsed_args.input_csv)
    smiles_list = df[parsed_args.smiles_col].tolist()

    featurizer = MolecularGraphFeaturizer()
    if parsed_args.model_path:
        model = load_model_from_checkpoint(parsed_args.model_path)
    else:
        model = MPNNClassifier(atomic_dim=featurizer.atom_dim, bond_dim=featurizer.bond_dim)

    model.eval()
    dataset = MolecularGraphDataset(smiles_list=smiles_list, featurizer=featurizer)
    if isinstance(model, DualChannelBBBPredictor):
        collate_fn = DualChannelCollate(tokenizer=model.seq_encoder.tokenizer)
    else:
        collate_fn = collate_molecular_graphs

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch_graph = {
                "node_features": batch["node_features"],
                "edge_features": batch["edge_features"],
                "connectivity_indices": batch["connectivity_indices"],
                "batch_assignment": batch["batch_assignment"],
            }
            if isinstance(model, DualChannelBBBPredictor):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                logits, _ = model(batch_graph, input_ids, attention_mask)
            else:
                logits, _ = model(batch_graph)
            prob = torch.sigmoid(logits).view(-1).tolist()
            probs.extend(prob)

    df["bbb_probability"] = probs
    df["bbb_prediction"] = [1 if p >= 0.5 else 0 for p in probs]
    output_path = Path(parsed_args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions successfully written to {output_path}")


def main() -> None:
    """Unified CLI router supporting subcommands train, eval, and predict."""
    parser = argparse.ArgumentParser(
        description="BBB-MolGraph Command-Line Interface suite.",
        usage="python -m bbb_molgraph.cli {train,eval,predict} [options]",
    )
    parser.add_argument("command", choices=["train", "eval", "predict"], help="Subcommand to execute.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Subcommand specific arguments.")

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)

    if sys.argv[1] in ["-h", "--help"]:
        parser.print_help()
        sys.exit(0)

    cmd = sys.argv[1]
    remaining = sys.argv[2:]

    if cmd == "train":
        train_cli(args=remaining)
    elif cmd == "eval":
        eval_cli(args=remaining)
    elif cmd == "predict":
        predict_cli(args=remaining)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
