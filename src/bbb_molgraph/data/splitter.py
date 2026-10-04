"""
Bemis-Murcko Scaffold Splitter for rigorous, leak-free molecular dataset partitioning.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from bbb_molgraph.utils.logging import logger


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold SMILES for a given molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold
    except Exception:
        return ""


class ScaffoldSplitter:
    """
    Splits molecular datasets based on Bemis-Murcko scaffolds to eliminate data leakage.
    Supports K-Fold Scaffold Cross-Validation and standard Train/Val/Test partitioning.
    """

    def __init__(self, include_chirality: bool = False) -> None:
        self.include_chirality = include_chirality

    def split_train_val_test(
        self,
        smiles_list: List[str],
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split indices into train, val, and test partitions based on scaffolds.
        """
        total = len(smiles_list)
        train_cutoff = int(train_frac * total)
        val_cutoff = int((train_frac + val_frac) * total)

        # Group indices by scaffold
        scaffold_to_indices = defaultdict(list)
        for idx, smi in enumerate(smiles_list):
            scaffold = generate_scaffold(smi, include_chirality=self.include_chirality)
            scaffold_to_indices[scaffold].append(idx)

        # Sort scaffolds by size descending for balanced grouping
        rng = np.random.RandomState(seed)
        scaffold_groups = list(scaffold_to_indices.values())
        # Shuffle with deterministic seed then sort by group length
        rng.shuffle(scaffold_groups)
        scaffold_groups.sort(key=len, reverse=True)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        for group in scaffold_groups:
            if len(train_indices) + len(group) <= train_cutoff:
                train_indices.extend(group)
            elif len(train_indices) + len(val_indices) + len(group) <= val_cutoff:
                val_indices.extend(group)
            else:
                test_indices.extend(group)

        return train_indices, val_indices, test_indices

    def k_fold_scaffold_cv(
        self,
        smiles_list: List[str],
        k: int = 5,
        seed: int = 42,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Partition dataset into K folds such that all molecules with the same scaffold
        belong exclusively to a single fold.

        Returns:
            List of (train_indices, test_indices) tuples for each fold.
        """
        scaffold_to_indices = defaultdict(list)
        for idx, smi in enumerate(smiles_list):
            scaffold = generate_scaffold(smi, include_chirality=self.include_chirality)
            scaffold_to_indices[scaffold].append(idx)

        rng = np.random.RandomState(seed)
        scaffold_groups = list(scaffold_to_indices.values())
        rng.shuffle(scaffold_groups)
        scaffold_groups.sort(key=len, reverse=True)

        folds: List[List[int]] = [[] for _ in range(k)]
        fold_sizes: List[int] = [0] * k

        for group in scaffold_groups:
            # Greedily assign to the fold with currently minimal size
            min_fold_idx = int(np.argmin(fold_sizes))
            folds[min_fold_idx].extend(group)
            fold_sizes[min_fold_idx] += len(group)

        cv_splits: List[Tuple[List[int], List[int]]] = []
        for i in range(k):
            test_indices = sorted(folds[i])
            train_indices = sorted([idx for j in range(k) if j != i for idx in folds[j]])
            cv_splits.append((train_indices, test_indices))

        return cv_splits

    @staticmethod
    def save_folds_to_json(
        folds: List[Tuple[List[int], List[int]]],
        filepath: Union[str, Path],
    ) -> None:
        """
        Save cross-validation splits to JSON format for rigorous reproducibility.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        serializable_data = [
            {"fold": i, "train": train_idx, "test": test_idx}
            for i, (train_idx, test_idx) in enumerate(folds)
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2)
        logger.info(f"Saved {len(folds)}-fold scaffold splits to {filepath}")

    @staticmethod
    def load_folds_from_json(filepath: Union[str, Path]) -> List[Tuple[List[int], List[int]]]:
        """
        Load pre-generated cross-validation splits from JSON file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Scaffold split file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        folds = [(item["train"], item["test"]) for item in data]
        return folds
