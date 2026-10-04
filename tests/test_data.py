"""
Unit tests for data featurization, datasets, and scaffold splitting.
"""

import pytest
import torch
from bbb_molgraph.data.dataset import MolecularGraphDataset, collate_molecular_graphs
from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer
from bbb_molgraph.data.splitter import ScaffoldSplitter, generate_scaffold


def test_featurizer_valid_smiles():
    featurizer = MolecularGraphFeaturizer()
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    graph = featurizer.smiles_to_graph(smiles)

    assert "node_features" in graph
    assert "edge_features" in graph
    assert "connectivity_indices" in graph
    assert "num_nodes" in graph

    num_atoms = graph["num_nodes"].item()
    assert num_atoms == 13
    assert graph["node_features"].shape == (13, featurizer.atom_dim)
    assert graph["edge_features"].shape[1] == featurizer.bond_dim
    assert graph["connectivity_indices"].shape[1] == 2


def test_featurizer_invalid_smiles():
    featurizer = MolecularGraphFeaturizer()
    with pytest.raises(ValueError):
        featurizer.smiles_to_graph("INVALID_SMILES_STRING_12345")


def test_featurizer_caching():
    featurizer = MolecularGraphFeaturizer(use_cache=True)
    smiles = "CCO"  # Ethanol
    g1 = featurizer.smiles_to_graph(smiles)
    g2 = featurizer.smiles_to_graph(smiles)
    assert g1 is g2
    featurizer.clear_cache()
    assert len(featurizer._cache) == 0


def test_scaffold_splitter_disjoint():
    smiles_list = [
        "c1ccccc1",         # Benzene
        "c1ccccc1O",        # Phenol
        "c1ccccc1C(=O)O",   # Benzoic acid
        "CC(C)CC",          # Alkane (no ring scaffold)
        "CCCC",             # Alkane (no ring scaffold)
        "c1ncccc1",         # Pyridine
    ]

    splitter = ScaffoldSplitter()
    folds = splitter.k_fold_scaffold_cv(smiles_list, k=2, seed=42)
    assert len(folds) == 2

    for train_idx, test_idx in folds:
        # Assert no overlap in indices
        assert set(train_idx).isdisjoint(set(test_idx))

        # Check scaffold disjointness
        train_scaffolds = {generate_scaffold(smiles_list[i]) for i in train_idx}
        test_scaffolds = {generate_scaffold(smiles_list[i]) for i in test_idx}
        # Scaffolds (excluding empty string for non-ring molecules) must be disjoint
        valid_train = {s for s in train_scaffolds if s}
        valid_test = {s for s in test_scaffolds if s}
        assert valid_train.isdisjoint(valid_test)


def test_dataset_and_collate():
    smiles_list = ["CCO", "c1ccccc1"]
    labels = [1.0, 0.0]
    featurizer = MolecularGraphFeaturizer()
    dataset = MolecularGraphDataset(smiles_list=smiles_list, labels=labels, featurizer=featurizer)
    assert len(dataset) == 2

    batch = [dataset[0], dataset[1]]
    collated = collate_molecular_graphs(batch)

    assert collated["batch_size"] == 2
    assert "node_features" in collated
    assert "edge_features" in collated
    assert "connectivity_indices" in collated
    assert "batch_assignment" in collated
    assert "labels" in collated
    assert collated["labels"].shape == (2, 1)
