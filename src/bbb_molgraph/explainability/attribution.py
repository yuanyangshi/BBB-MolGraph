"""
Attention attribution extraction from dual-channel multimodal architectures.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer


class DualChannelAttributionExtractor:
    """
    Extracts atom-level and sequence-token-level attention attributions for a given molecule.
    """

    def __init__(self, model: nn.Module, featurizer: Optional[MolecularGraphFeaturizer] = None) -> None:
        self.model = model
        self.featurizer = featurizer or MolecularGraphFeaturizer()
        self.model.eval()

    @torch.no_grad()
    def explain_smiles(
        self,
        smiles: str,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Explain the model's prediction on a single SMILES string.

        Returns:
            Dictionary containing:
            - 'probability': predicted BBB+ probability (0.0 to 1.0)
            - 'atom_attentions': 1D numpy array of normalized weights per atom (summing to 1.0)
            - 'token_attentions': optional token attention array if tokenizer provided
            - 'smiles': the input SMILES
            - 'num_atoms': count of atoms
        """
        device = device or next(self.model.parameters()).device
        graph = self.featurizer.smiles_to_graph(smiles)

        num_atoms = graph["node_features"].size(0)
        batch_graph = {
            "node_features": graph["node_features"].to(device),
            "edge_features": graph["edge_features"].to(device),
            "connectivity_indices": graph["connectivity_indices"].to(device),
            "batch_assignment": torch.zeros(num_atoms, dtype=torch.long, device=device),
        }

        token_attns_np = None

        if tokenizer is not None and hasattr(self.model, "seq_encoder"):
            tokens = tokenizer(
                [smiles],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            logits, attributions = self.model(batch_graph, input_ids, attention_mask)
            if attributions.get("seq_attentions") is not None:
                token_attns_np = attributions["seq_attentions"].cpu().numpy().ravel()
        elif hasattr(self.model, "seq_encoder"):
            # Model is dual-channel but no tokenizer passed, construct dummy tokens
            dummy_ids = torch.zeros((1, 10), dtype=torch.long, device=device)
            dummy_mask = torch.ones((1, 10), dtype=torch.long, device=device)
            logits, attributions = self.model(batch_graph, dummy_ids, dummy_mask)
        else:
            # Single-channel graph model
            logits, attributions = self.model(batch_graph)

        prob = float(torch.sigmoid(logits).cpu().view(-1)[0].item())

        if isinstance(attributions, dict):
            atom_attns = attributions.get("graph_attentions", None)
        else:
            atom_attns = attributions
        if isinstance(atom_attns, torch.Tensor):
            atom_attns_np = atom_attns.cpu().numpy().ravel()
        else:
            atom_attns_np = np.ones(num_atoms, dtype=np.float32)

        # Min-max normalization for visualization contrast
        if atom_attns_np.max() > atom_attns_np.min():
            norm_attns = (atom_attns_np - atom_attns_np.min()) / (
                atom_attns_np.max() - atom_attns_np.min()
            )
        else:
            norm_attns = np.ones_like(atom_attns_np)

        return {
            "smiles": smiles,
            "probability": prob,
            "atom_attentions": norm_attns,
            "raw_atom_attentions": atom_attns_np,
            "token_attentions": token_attns_np,
            "num_atoms": num_atoms,
        }
