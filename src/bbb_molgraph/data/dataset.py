"""
PyTorch Dataset classes and custom collate functions for molecular graphs and dual-channel inputs.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer


class MolecularGraphDataset(Dataset):
    """
    Dataset storing precomputed or on-the-fly molecular graphs along with targets.
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
        featurizer: Optional[MolecularGraphFeaturizer] = None,
        precomputed_graphs: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.smiles_list = smiles_list
        self.labels = labels
        self.featurizer = featurizer or MolecularGraphFeaturizer()
        self.precomputed_graphs = precomputed_graphs

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        smiles = self.smiles_list[idx]

        if self.precomputed_graphs is not None:
            graph_dict = self.precomputed_graphs[idx]
        else:
            graph_dict = self.featurizer.smiles_to_graph(smiles)

        item: Dict[str, Any] = {
            "smiles": smiles,
            "node_features": graph_dict["node_features"],
            "edge_features": graph_dict["edge_features"],
            "connectivity_indices": graph_dict["connectivity_indices"],
            "num_nodes": graph_dict["num_nodes"],
        }

        if self.labels is not None:
            item["label"] = float(self.labels[idx])

        return item


def collate_molecular_graphs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate variable-sized molecular graphs into a single contiguous batch.
    
    Offsets connectivity indices by cumulative atom counts and constructs
    a batch assignment tensor mapping each atom to its graph index.
    """
    atom_list: List[torch.Tensor] = []
    bond_list: List[torch.Tensor] = []
    connectivity_list: List[torch.Tensor] = []
    batch_assignment_list: List[torch.Tensor] = []
    smiles_list: List[str] = []
    labels_list: List[float] = []

    current_atom_offset = 0

    for i, item in enumerate(batch):
        num_atoms = item["node_features"].size(0)
        atom_list.append(item["node_features"])
        bond_list.append(item["edge_features"])

        # Offset node indices in connectivity pairs
        offset_conn = item["connectivity_indices"] + current_atom_offset
        connectivity_list.append(offset_conn)

        # Batch assignment index for graph pooling
        batch_assignment_list.append(
            torch.full((num_atoms,), i, dtype=torch.long)
        )

        smiles_list.append(item["smiles"])
        if "label" in item:
            labels_list.append(item["label"])

        current_atom_offset += num_atoms

    collated: Dict[str, Any] = {
        "node_features": torch.cat(atom_list, dim=0),
        "edge_features": torch.cat(bond_list, dim=0),
        "connectivity_indices": torch.cat(connectivity_list, dim=0),
        "batch_assignment": torch.cat(batch_assignment_list, dim=0),
        "smiles": smiles_list,
        "batch_size": len(batch),
    }

    if labels_list:
        collated["labels"] = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)

    return collated


class DualChannelCollate:
    """
    Collate function combining batched molecular graphs with tokenized sequences.
    """

    def __init__(self, tokenizer: Any, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = collate_molecular_graphs(batch)

        # Tokenize SMILES strings for the sequence branch
        encoded = self.tokenizer(
            collated["smiles"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        collated["input_ids"] = encoded["input_ids"]
        collated["attention_mask"] = encoded["attention_mask"]
        return collated
