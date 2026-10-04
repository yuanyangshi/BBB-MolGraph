"""
Dual-Channel Multimodal Framework combining MPNN topology encoding with MolFormer sequence modeling.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from bbb_molgraph.core.registry import MODELS
from bbb_molgraph.models.molformer import MolFormerEncoder
from bbb_molgraph.models.mpnn import MPNNEncoder
from bbb_molgraph.utils.logging import logger


class GatedMultimodalFusion(nn.Module):
    """
    Learns dynamic gating weights to selectively balance graph and sequence modalities.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
        )

    def forward(self, h_graph: torch.Tensor, h_seq: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([h_graph, h_seq], dim=-1)
        z = self.proj(concat)
        g = self.gate(concat)
        fused = g * h_graph + (1.0 - g) * h_seq + z
        return fused


@MODELS.register("dual_channel_fusion")
class DualChannelBBBPredictor(nn.Module):
    """
    Proposed Dual-Channel Multimodal BBB Penetration Predictor.
    Integrates MPNN graph topological representations with MolFormer sequence embeddings.
    """

    def __init__(
        self,
        atomic_dim: int = 30,
        bond_dim: int = 11,
        mpnn_hidden_dim: int = 64,
        propagation_steps: int = 4,
        molformer_model_name: str = "ibm-research/MoLFormer-XL-both-10pct",
        molformer_out_dim: int = 64,
        freeze_molformer: bool = True,
        use_fallback_molformer: bool = False,
        fusion_mode: str = "gated",
        fusion_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode

        self.graph_encoder = MPNNEncoder(
            atomic_dim=atomic_dim,
            bond_dim=bond_dim,
            hidden_dim=mpnn_hidden_dim,
            propagation_steps=propagation_steps,
            dropout=dropout,
        )

        self.seq_encoder = MolFormerEncoder(
            model_name=molformer_model_name,
            out_dim=molformer_out_dim,
            freeze=freeze_molformer,
            use_fallback=use_fallback_molformer,
        )

        # Ensure feature dimensions match for gated fusion
        if fusion_mode == "gated":
            self.graph_proj = (
                nn.Linear(mpnn_hidden_dim, fusion_dim)
                if mpnn_hidden_dim != fusion_dim
                else nn.Identity()
            )
            self.seq_proj = (
                nn.Linear(molformer_out_dim, fusion_dim)
                if molformer_out_dim != fusion_dim
                else nn.Identity()
            )
            self.fusion_layer = GatedMultimodalFusion(dim=fusion_dim)
            clf_in_dim = fusion_dim
        else:
            # Standard concatenation fusion
            self.fusion_layer = nn.Sequential(
                nn.Linear(mpnn_hidden_dim + molformer_out_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            clf_in_dim = fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, clf_in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_in_dim // 2, 1),
        )

    def forward(
        self,
        batch_graph: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass producing logits and dual-channel attention attributions.

        Returns:
            logits: (batch_size, 1)
            attributions: dict containing 'graph_attentions' and 'seq_attentions'
        """
        h_graph, graph_attn = self.graph_encoder(batch_graph)
        h_seq, seq_attn = self.seq_encoder(input_ids, attention_mask)

        if self.fusion_mode == "gated":
            h_g = self.graph_proj(h_graph)
            h_s = self.seq_proj(h_seq)
            h_fused = self.fusion_layer(h_g, h_s)
        else:
            h_combined = torch.cat([h_graph, h_seq], dim=-1)
            h_fused = self.fusion_layer(h_combined)

        logits = self.classifier(h_fused)

        attributions = {
            "graph_attentions": graph_attn,
            "seq_attentions": seq_attn,
        }
        return logits, attributions

    @classmethod
    def from_pretrained(cls, checkpoint_path: Union[str, Path], **kwargs: Any) -> "DualChannelBBBPredictor":
        """
        Load model architecture and weights from a local checkpoint file with shape inference.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_config = checkpoint.get("config", {})
        model_config.update(kwargs)

        # Infer topology dimensions
        edge_weight_key = "graph_encoder.propagation.bond_processor.edge_transformation.weight"
        if edge_weight_key in state_dict:
            if "atomic_dim" not in model_config:
                model_config["atomic_dim"] = state_dict[edge_weight_key].shape[0]
            if "bond_dim" not in model_config:
                model_config["bond_dim"] = state_dict[edge_weight_key].shape[1]

        gru_weight_key = "graph_encoder.propagation.gru_cell.weight_hh"
        if gru_weight_key in state_dict and "mpnn_hidden_dim" not in model_config:
            model_config["mpnn_hidden_dim"] = state_dict[gru_weight_key].shape[1]

        # Infer seq_encoder out dimension
        if "molformer_out_dim" not in model_config and "seq_encoder.seq_proj.0.weight" in state_dict:
            model_config["molformer_out_dim"] = state_dict["seq_encoder.seq_proj.0.weight"].shape[0]

        # Infer fusion dimension
        if "fusion_dim" not in model_config:
            if "seq_proj.weight" in state_dict:
                model_config["fusion_dim"] = state_dict["seq_proj.weight"].shape[0]
            elif "graph_proj.weight" in state_dict:
                model_config["fusion_dim"] = state_dict["graph_proj.weight"].shape[0]
            elif "fusion_layer.gate.0.weight" in state_dict:
                model_config["fusion_dim"] = state_dict["fusion_layer.gate.0.weight"].shape[0]

        # Infer fallback molformer if state dict contains lightweight layers
        if "use_fallback_molformer" not in model_config:
            is_fallback = any(
                "seq_encoder.backbone.embedding." in k
                or "seq_encoder.backbone.transformer." in k
                or "seq_encoder.embedding." in k
                for k in state_dict.keys()
            )
            if is_fallback:
                model_config["use_fallback_molformer"] = True

        model = cls(**model_config)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded pretrained model weights from {checkpoint_path}")
        return model
