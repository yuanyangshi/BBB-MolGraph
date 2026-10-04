"""
Unit tests for BBB-MolGraph Command-Line Interface (CLI).
"""

from pathlib import Path
import sys
import pandas as pd
import pytest
import torch

from bbb_molgraph.cli import eval_cli, main, predict_cli
from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer
from bbb_molgraph.models.mpnn import MPNNClassifier


@pytest.fixture
def dummy_checkpoint_and_data(tmp_path: Path):
    # Create tiny dummy model
    featurizer = MolecularGraphFeaturizer()
    model = MPNNClassifier(
        atomic_dim=featurizer.atom_dim,
        bond_dim=featurizer.bond_dim,
        hidden_dim=16,
    )
    ckpt_path = tmp_path / "dummy_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "atomic_dim": featurizer.atom_dim,
                "bond_dim": featurizer.bond_dim,
                "hidden_dim": 16,
            },
        },
        ckpt_path,
    )

    # Create dummy data CSV
    df = pd.DataFrame(
        {
            "name": ["Aspirin", "Ethanol", "Benzene", "AceticAcid"],
            "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1", "CC(=O)O"],
            "label": [1, 1, 0, 0],
        }
    )
    csv_path = tmp_path / "dummy_data.csv"
    df.to_csv(csv_path, index=False)

    return ckpt_path, csv_path


def test_cli_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bbb_molgraph.cli", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_cli_predict(dummy_checkpoint_and_data, tmp_path):
    ckpt_path, csv_path = dummy_checkpoint_and_data
    out_csv = tmp_path / "nested_dir" / "preds.csv"

    args = [
        "--input-csv",
        str(csv_path),
        "--output-csv",
        str(out_csv),
        "--smiles-col",
        "smiles",
        "--model-path",
        str(ckpt_path),
    ]

    predict_cli(args=args)

    assert out_csv.exists()
    df_out = pd.read_csv(out_csv)
    assert "bbb_probability" in df_out.columns
    assert "bbb_prediction" in df_out.columns
    assert len(df_out) == 4
    assert all(0.0 <= p <= 1.0 for p in df_out["bbb_probability"])


def test_cli_eval(dummy_checkpoint_and_data):
    ckpt_path, csv_path = dummy_checkpoint_and_data
    args = [
        "--model-path",
        str(ckpt_path),
        "--data-csv",
        str(csv_path),
    ]

    # Should execute and print metrics without raising errors
    eval_cli(args=args)
