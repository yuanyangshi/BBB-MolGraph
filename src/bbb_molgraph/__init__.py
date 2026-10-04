"""
BBB-MolGraph: A Publication-Grade Deep Learning Framework for Blood-Brain Barrier Permeability Prediction.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer
from bbb_molgraph.data.splitter import ScaffoldSplitter
from bbb_molgraph.evaluation.bootstrap import bootstrap_confidence_interval
from bbb_molgraph.evaluation.metrics import compute_classification_metrics
from bbb_molgraph.explainability.attribution import DualChannelAttributionExtractor
from bbb_molgraph.explainability.visualizer import PharmacophoreVisualizer
from bbb_molgraph.models import (
    DualChannelBBBPredictor,
    MPNNClassifier,
    MPNNEncoder,
    MolFormerEncoder,
    load_model_from_checkpoint,
)
from bbb_molgraph.training.trainer import Trainer
from bbb_molgraph.utils.logging import logger, setup_logger
from bbb_molgraph.utils.seed import set_deterministic_seed

__version__ = "1.0.0"


def load_pretrained_model(
    checkpoint_path_or_name: Union[str, Path] = "default",
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Load a pre-trained BBB-MolGraph predictor.

    Args:
        checkpoint_path_or_name: Filepath to model checkpoint or named preset.
        device: PyTorch device to place model on.

    Returns:
        Loaded PyTorch model in eval mode.
    """
    path = Path(checkpoint_path_or_name)
    featurizer = MolecularGraphFeaturizer()

    if path.exists():
        model = load_model_from_checkpoint(path, device=device)
    else:
        logger.info(
            f"Checkpoint '{checkpoint_path_or_name}' not found locally. "
            "Initializing fresh DualChannelBBBPredictor architecture."
        )
        model = DualChannelBBBPredictor(
            atomic_dim=featurizer.atom_dim,
            bond_dim=featurizer.bond_dim,
            use_fallback_molformer=True,
        )
        if device is not None:
            model = model.to(device)

    model.eval()
    return model


def predict_smiles(
    smiles: str,
    model: Optional[torch.nn.Module] = None,
    explain: bool = False,
    device: Optional[torch.device] = None,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Predict Blood-Brain Barrier (BBB) permeability for a single SMILES string.

    Args:
        smiles: Valid SMILES string representing the molecule.
        model: Optional instantiated model. If None, loads default model.
        explain: If True, computes atom-level attention attributions.
        device: Optional torch.device.

    Returns:
        If explain is False: probability of BBB permeability (float, 0.0 to 1.0)
        If explain is True: tuple of (probability, explanation_dict)
    """
    if model is None:
        model = load_pretrained_model(device=device)

    extractor = DualChannelAttributionExtractor(model=model)
    res = extractor.explain_smiles(smiles=smiles, device=device)
    prob = res["probability"]

    if explain:
        return prob, res
    return prob


__all__ = [
    "__version__",
    "load_pretrained_model",
    "predict_smiles",
    "set_deterministic_seed",
    "MolecularGraphFeaturizer",
    "ScaffoldSplitter",
    "MPNNEncoder",
    "MPNNClassifier",
    "MolFormerEncoder",
    "DualChannelBBBPredictor",
    "Trainer",
    "compute_classification_metrics",
    "bootstrap_confidence_interval",
    "PharmacophoreVisualizer",
    "setup_logger",
    "logger",
]
