from pathlib import Path
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from bbb_molgraph.models.baselines import GCNBaseline, TabularMLBaseline
from bbb_molgraph.models.fusion import DualChannelBBBPredictor, GatedMultimodalFusion
from bbb_molgraph.models.molformer import MolFormerClassifier, MolFormerEncoder
from bbb_molgraph.models.mpnn import MPNNClassifier, MPNNEncoder
from bbb_molgraph.utils.logging import logger


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Intelligently infer model architecture (DualChannel vs MPNN) from checkpoint and instantiate.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Check state_dict keys to identify architecture
    is_dual = any("graph_encoder." in k or "seq_encoder." in k for k in state_dict.keys())

    if is_dual:
        model = DualChannelBBBPredictor.from_pretrained(path, **kwargs)
    else:
        model = MPNNClassifier.from_pretrained(path, **kwargs)

    if device is not None:
        model = model.to(device)

    model.eval()
    return model


__all__ = [
    "MPNNEncoder",
    "MPNNClassifier",
    "MolFormerEncoder",
    "MolFormerClassifier",
    "DualChannelBBBPredictor",
    "GatedMultimodalFusion",
    "TabularMLBaseline",
    "GCNBaseline",
    "load_model_from_checkpoint",
]

