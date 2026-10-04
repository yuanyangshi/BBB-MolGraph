"""
Molecular Graph Featurizer converting SMILES representations to graph topology tensors.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from rdkit import Chem, RDLogger

from bbb_molgraph.core.constants import (
    DEFAULT_PERMITTED_ATOM_FEATURES,
    DEFAULT_PERMITTED_BOND_FEATURES,
)

# Suppress minor RDKit logs during standard parsing
RDLogger.DisableLog("rdApp.*")


class MolecularPropertyEncoder:
    """
    One-hot encoder for categorical chemical properties.
    """

    def __init__(self, permitted_value_sets: Dict[str, set]) -> None:
        self.total_dimensions = 0
        self.property_mapping_dict: Dict[str, Dict[Any, int]] = {}

        for property_identifier, permitted_values in permitted_value_sets.items():
            ordered_values = sorted(list(permitted_values), key=lambda x: str(x))
            index_mapping = {
                val: self.total_dimensions + i
                for i, val in enumerate(ordered_values)
            }
            self.property_mapping_dict[property_identifier] = index_mapping
            self.total_dimensions += len(ordered_values)

    def transform_to_vector(self, molecular_component) -> np.ndarray:
        encoding_vector = np.zeros((self.total_dimensions,), dtype=np.float32)
        for property_identifier, index_mapping in self.property_mapping_dict.items():
            extracted_property = getattr(self, property_identifier)(molecular_component)
            if extracted_property in index_mapping:
                encoding_vector[index_mapping[extracted_property]] = 1.0
        return encoding_vector


class AtomicPropertyEncoder(MolecularPropertyEncoder):
    """Encodes atomic properties (symbol, valence, hydrogen count, hybridization)."""

    def symbol(self, atom: Chem.Atom) -> str:
        return atom.GetSymbol()

    def n_valence(self, atom: Chem.Atom) -> int:
        return atom.GetTotalValence()

    def n_hydrogens(self, atom: Chem.Atom) -> int:
        return atom.GetTotalNumHs()

    def hybridization(self, atom: Chem.Atom) -> str:
        return atom.GetHybridization().name.lower()


class BondPropertyEncoder(MolecularPropertyEncoder):
    """Encodes chemical bond properties (type, conjugation) with a dedicated self-loop feature."""

    def __init__(self, permitted_value_sets: Dict[str, set]) -> None:
        super().__init__(permitted_value_sets)
        # Allocate one additional dimension at the end for self-loop / no-bond state
        self.total_dimensions += 1

    def transform_to_vector(self, chemical_bond: Optional[Chem.Bond]) -> np.ndarray:
        encoding_vector = np.zeros((self.total_dimensions,), dtype=np.float32)
        if chemical_bond is None:
            # Self-connection / dummy bond token
            encoding_vector[-1] = 1.0
            return encoding_vector

        for property_identifier, index_mapping in self.property_mapping_dict.items():
            extracted_property = getattr(self, property_identifier)(chemical_bond)
            if extracted_property in index_mapping:
                encoding_vector[index_mapping[extracted_property]] = 1.0
        return encoding_vector

    def bond_type(self, bond: Chem.Bond) -> str:
        return bond.GetBondType().name.lower()

    def conjugated(self, bond: Chem.Bond) -> bool:
        return bond.GetIsConjugated()


class MolecularGraphFeaturizer:
    """
    Production-grade featurizer converting SMILES strings into graph tensors.
    """

    def __init__(
        self,
        permitted_atom_features: Optional[Dict[str, set]] = None,
        permitted_bond_features: Optional[Dict[str, set]] = None,
        use_cache: bool = True,
    ) -> None:
        atom_features = permitted_atom_features or DEFAULT_PERMITTED_ATOM_FEATURES
        bond_features = permitted_bond_features or DEFAULT_PERMITTED_BOND_FEATURES

        self.atom_encoder = AtomicPropertyEncoder(atom_features)
        self.bond_encoder = BondPropertyEncoder(bond_features)
        self.atom_dim = self.atom_encoder.total_dimensions
        self.bond_dim = self.bond_encoder.total_dimensions

        self.use_cache = use_cache
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
        """Convert a SMILES string to a sanitized RDKit molecule."""
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        clean_flag = Chem.SanitizeMol(mol, catchErrors=True)
        if clean_flag != Chem.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ clean_flag)

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return mol

    def smiles_to_graph(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Extract graph representations from a single SMILES string.

        Returns:
            Dictionary containing:
            - 'node_features': Tensor of shape (num_atoms, atom_dim)
            - 'edge_features': Tensor of shape (num_edges, bond_dim)
            - 'connectivity_indices': Tensor of shape (num_edges, 2)
            - 'num_nodes': int tensor
        """
        if self.use_cache and smiles in self._cache:
            return self._cache[smiles]

        mol = self.smiles_to_mol(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse and sanitize SMILES string: '{smiles}'")

        atom_vectors: List[np.ndarray] = []
        bond_vectors: List[np.ndarray] = []
        connectivity_pairs: List[List[int]] = []

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            raise ValueError(f"SMILES string has 0 atoms: '{smiles}'")

        for atom in mol.GetAtoms():
            atom_vectors.append(self.atom_encoder.transform_to_vector(atom))
            idx = atom.GetIdx()

            # Add self-connection (self-loop) with dummy bond encoding
            connectivity_pairs.append([idx, idx])
            bond_vectors.append(self.bond_encoder.transform_to_vector(None))

            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                bond = mol.GetBondBetweenAtoms(idx, neighbor_idx)
                connectivity_pairs.append([idx, neighbor_idx])
                bond_vectors.append(self.bond_encoder.transform_to_vector(bond))

        graph_dict = {
            "node_features": torch.tensor(np.array(atom_vectors, dtype=np.float32), dtype=torch.float32),
            "edge_features": torch.tensor(np.array(bond_vectors, dtype=np.float32), dtype=torch.float32),
            "connectivity_indices": torch.tensor(np.array(connectivity_pairs, dtype=np.int64), dtype=torch.long),
            "num_nodes": torch.tensor(num_atoms, dtype=torch.long),
        }

        if self.use_cache:
            self._cache[smiles] = graph_dict

        return graph_dict

    def clear_cache(self) -> None:
        """Clear cached graph representations."""
        self._cache.clear()
