"""
Message Passing Neural Network (MPNN) Encoder and Classifier for molecular graph topology.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from bbb_molgraph.core.registry import MODELS
from bbb_molgraph.modules.message_passing import IterativeMessagePropagation
from bbb_molgraph.modules.readout import AttentionBasedMolecularReadout, MeanReadout
from bbb_molgraph.utils.logging import logger


@MODELS.register("mpnn_encoder")
class MPNNEncoder(nn.Module):
    """
    MPNN Graph Encoder consisting of message propagation and attention readout.
    """

    def __init__(
        self,
        atomic_dim: int = 30,
        bond_dim: int = 11,
        hidden_dim: int = 64,
        propagation_steps: int = 4,
        readout_type: str = "attention",
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.atomic_dim = atomic_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim

        self.propagation = IterativeMessagePropagation(
            atomic_dim=atomic_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            propagation_steps=propagation_steps,
            dropout=dropout,
        )

        if readout_type == "attention":
            self.readout = AttentionBasedMolecularReadout(
                embedding_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim * 4,
                dropout=dropout,
            )
        else:
            self.readout = MeanReadout(embedding_dim=hidden_dim)

    def forward(
        self,
        batch_graph: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_graph: Dictionary containing 'node_features', 'edge_features',
                         'connectivity_indices', and 'batch_assignment'.

        Returns:
            graph_embeddings: (batch_size, hidden_dim)
            atom_attentions: (total_atoms,)
        """
        node_features = batch_graph["node_features"]
        edge_features = batch_graph["edge_features"]
        connectivity = batch_graph["connectivity_indices"]
        batch_assignment = batch_graph["batch_assignment"]

        updated_nodes = self.propagation(node_features, edge_features, connectivity)
        graph_embeddings, atom_attentions = self.readout(updated_nodes, batch_assignment)

        return graph_embeddings, atom_attentions


@MODELS.register("mpnn_classifier")
class MPNNClassifier(nn.Module):
    """
    End-to-end MPNN classification baseline for single-modality benchmarking.
    """

    def __init__(
        self,
        atomic_dim: int = 30,
        bond_dim: int = 11,
        hidden_dim: int = 64,
        propagation_steps: int = 4,
        readout_type: str = "attention",
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = MPNNEncoder(
            atomic_dim=atomic_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            propagation_steps=propagation_steps,
            readout_type=readout_type,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch_graph: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, atom_attentions = self.encoder(batch_graph)
        logits = self.classifier(embeddings)
        return logits, atom_attentions

    @classmethod
    def from_pretrained(cls, checkpoint_path: Union[str, Path], **kwargs: Any) -> "MPNNClassifier":
        """
        Load MPNNClassifier architecture and weights with automatic shape inference.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        config = checkpoint.get("config", {})
        config.update(kwargs)

        # Infer dimensions from weights if not explicitly defined
        if "hidden_dim" not in config:
            if "encoder.propagation.gru_cell.weight_hh" in state_dict:
                config["hidden_dim"] = state_dict["encoder.propagation.gru_cell.weight_hh"].shape[1]

        edge_weight_key = "encoder.propagation.bond_processor.edge_transformation.weight"
        if edge_weight_key in state_dict:
            if "atomic_dim" not in config:
                config["atomic_dim"] = state_dict[edge_weight_key].shape[0]
            if "bond_dim" not in config:
                config["bond_dim"] = state_dict[edge_weight_key].shape[1]

        model = cls(**config)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded MPNNClassifier weights from {path}")
        return model
