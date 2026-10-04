"""
Readout and graph pooling mechanisms with attention extraction for interpretability.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBasedMolecularReadout(nn.Module):
    """
    Self-attention molecular readout mechanism aggregating node features into a graph embedding.
    Extracts attention weights across atoms for downstream explainability and 2D heatmaps.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        batch_assignment: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (total_atoms, embedding_dim)
            batch_assignment: (total_atoms,) with graph index for each atom

        Returns:
            graph_embeddings: (batch_size, embedding_dim)
            atom_attentions: (total_atoms,) attention weights per atom
        """
        batch_size = int(batch_assignment.max().item()) + 1
        separated_nodes = []
        atom_counts = []

        for i in range(batch_size):
            mask = batch_assignment == i
            mol_nodes = node_features[mask]
            separated_nodes.append(mol_nodes)
            atom_counts.append(mol_nodes.size(0))

        max_atoms = max(atom_counts) if atom_counts else 1
        feature_dim = node_features.size(1)

        padded_nodes = []
        key_padding_masks = []

        for mol_nodes in separated_nodes:
            n = mol_nodes.size(0)
            if n < max_atoms:
                pad_tensor = torch.zeros(
                    max_atoms - n,
                    feature_dim,
                    device=node_features.device,
                    dtype=node_features.dtype,
                )
                padded = torch.cat([mol_nodes, pad_tensor], dim=0)
                mask = torch.tensor(
                    [False] * n + [True] * (max_atoms - n),
                    device=node_features.device,
                    dtype=torch.bool,
                )
            else:
                padded = mol_nodes
                mask = torch.zeros(n, device=node_features.device, dtype=torch.bool)

            padded_nodes.append(padded)
            key_padding_masks.append(mask)

        batched_input = torch.stack(padded_nodes, dim=0)  # (batch_size, max_atoms, dim)
        padding_mask = torch.stack(key_padding_masks, dim=0)  # (batch_size, max_atoms)

        # Multi-head attention with attention weights returned
        attn_out, attn_weights = self.mha(
            batched_input,
            batched_input,
            batched_input,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # Residual + Norm
        x = self.norm1(batched_input + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Masked average pooling across atom tokens to form graph embedding
        valid_mask = (~padding_mask).unsqueeze(-1).float()  # (batch_size, max_atoms, 1)
        sum_pooled = (x * valid_mask).sum(dim=1)
        lengths = valid_mask.sum(dim=1).clamp(min=1.0)
        graph_embeddings = sum_pooled / lengths

        # Extract per-atom attention attribution
        # attn_weights shape: (batch_size, max_atoms, max_atoms)
        # Average attention received by each atom token:
        avg_atom_attn = (attn_weights * valid_mask.transpose(1, 2)).sum(dim=1)  # (batch_size, max_atoms)
        # Flatten back into contiguous 1D tensor matching batch_assignment
        flattened_attentions = []
        for i, n in enumerate(atom_counts):
            flattened_attentions.append(avg_atom_attn[i, :n])

        atom_attentions = torch.cat(flattened_attentions, dim=0)

        return graph_embeddings, atom_attentions


class MeanReadout(nn.Module):
    """
    Standard mean pooling readout baseline.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(
        self,
        node_features: torch.Tensor,
        batch_assignment: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch_assignment.max().item()) + 1
        num_atoms = node_features.size(0)

        pooled = torch.zeros(
            batch_size,
            self.embedding_dim,
            device=node_features.device,
            dtype=node_features.dtype,
        )
        counts = torch.zeros(batch_size, 1, device=node_features.device, dtype=torch.float32)

        index_expanded = batch_assignment.unsqueeze(-1).expand(-1, self.embedding_dim)
        pooled.scatter_add_(0, index_expanded, node_features)

        counts.scatter_add_(0, batch_assignment.unsqueeze(-1), torch.ones(num_atoms, 1, device=node_features.device))
        counts = counts.clamp(min=1.0)
        graph_embeddings = pooled / counts

        # Dummy uniform attention for interface consistency
        dummy_attentions = torch.ones(num_atoms, device=node_features.device)
        return graph_embeddings, dummy_attentions
