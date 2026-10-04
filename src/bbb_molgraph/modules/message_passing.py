"""
Core message passing neural network layers and edge-conditioned propagation operators.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BondInformationProcessor(nn.Module):
    """
    Transforms bond representations and aggregates neighborhood messages.
    """

    def __init__(self, atomic_dim: int, bond_dim: int) -> None:
        super().__init__()
        self.atomic_dim = atomic_dim
        self.bond_dim = bond_dim
        self.edge_transformation = nn.Linear(bond_dim, atomic_dim)

    def forward(
        self,
        atomic_features: torch.Tensor,
        bond_features: torch.Tensor,
        connectivity_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate neighbor messages using transformed bond features.

        Args:
            atomic_features: (num_atoms, atomic_dim)
            bond_features: (num_edges, bond_dim)
            connectivity_indices: (num_edges, 2) where col 0 is target, col 1 is source

        Returns:
            aggregated_messages: (num_atoms, atomic_dim)
        """
        # Transform edge/bond features to match atomic feature space
        transformed_bonds = self.edge_transformation(bond_features)

        # Gather source atom features for each edge
        source_atom_indices = connectivity_indices[:, 1]
        neighbor_atoms = atomic_features[source_atom_indices]

        # Edge-conditioned interaction (Hadamard product)
        edge_messages = transformed_bonds * neighbor_atoms

        # Aggregate messages into target atoms using scatter_add
        target_atom_indices = connectivity_indices[:, 0]
        num_atoms = atomic_features.size(0)

        aggregated = torch.zeros(
            num_atoms,
            self.atomic_dim,
            device=atomic_features.device,
            dtype=atomic_features.dtype,
        )
        index_expanded = target_atom_indices.unsqueeze(-1).expand(-1, self.atomic_dim)
        aggregated.scatter_add_(0, index_expanded, edge_messages)

        return aggregated


class IterativeMessagePropagation(nn.Module):
    """
    Message passing network with iterative GRU state updates over multiple propagation steps.
    """

    def __init__(
        self,
        atomic_dim: int,
        bond_dim: int,
        hidden_dim: int = 64,
        propagation_steps: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.atomic_dim = atomic_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.propagation_steps = propagation_steps

        self.bond_processor = BondInformationProcessor(atomic_dim, bond_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Linear projection if initial atomic_dim differs from hidden_dim
        if atomic_dim != hidden_dim:
            self.input_projection = nn.Linear(atomic_dim, hidden_dim)
            self.msg_projection = nn.Linear(atomic_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()
            self.msg_projection = nn.Identity()

    def forward(
        self,
        atomic_features: torch.Tensor,
        bond_features: torch.Tensor,
        connectivity_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform multiple iterations of message passing.

        Returns:
            Updated node states of shape (num_atoms, hidden_dim).
        """
        h = self.input_projection(atomic_features)

        for _ in range(self.propagation_steps):
            # Message aggregation using current atomic representations
            # If input_projection was used, map back or aggregate directly
            messages = self.bond_processor(atomic_features, bond_features, connectivity_indices)
            messages = self.msg_projection(messages)
            messages = self.dropout(messages)

            # GRU update
            h = self.gru_cell(messages, h)

        return h
