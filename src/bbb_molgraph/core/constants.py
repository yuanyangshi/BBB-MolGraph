"""
Constants and standard chemical vocabularies for molecular graph encoding.
"""

from typing import Dict, Set

# Standard atom symbols permitted in BBBP dataset and drug-like molecules
PERMITTED_ATOM_SYMBOLS: Set[str] = {
    "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"
}

# Permitted valence states
PERMITTED_VALENCE_STATES: Set[int] = {0, 1, 2, 3, 4, 5, 6}

# Permitted hydrogen counts
PERMITTED_NUM_HYDROGENS: Set[int] = {0, 1, 2, 3, 4}

# Permitted atom hybridization states
PERMITTED_HYBRIDIZATION_STATES: Set[str] = {"s", "sp", "sp2", "sp3", "sp3d", "sp3d2"}

# Permitted bond types
PERMITTED_BOND_TYPES: Set[str] = {"single", "double", "triple", "aromatic"}

# Permitted bond stereochemical configurations
PERMITTED_BOND_STEREO: Set[str] = {"steroe_none", "stereo_any", "stereo_z", "stereo_e", "stereo_cis", "stereo_trans"}

# Default permitted feature configuration dictionary for featurizer
DEFAULT_PERMITTED_ATOM_FEATURES: Dict[str, Set] = {
    "symbol": PERMITTED_ATOM_SYMBOLS,
    "n_valence": PERMITTED_VALENCE_STATES,
    "n_hydrogens": PERMITTED_NUM_HYDROGENS,
    "hybridization": {"s", "sp", "sp2", "sp3"},
}

DEFAULT_PERMITTED_BOND_FEATURES: Dict[str, Set] = {
    "bond_type": PERMITTED_BOND_TYPES,
    "conjugated": {True, False},
}

# Default MolFormer model identifier on Hugging Face hub
DEFAULT_MOLFORMER_MODEL_NAME: str = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_EMBEDDING_DIM: int = 768

# Default target column and task definitions
DEFAULT_TARGET_COLUMN: str = "permeability_target"
DEFAULT_SMILES_COLUMN: str = "smiles"
