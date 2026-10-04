"""
Unit tests for explainability attribution extraction and 2D visualizer.
"""

import numpy as np
import pytest
from bbb_molgraph import load_pretrained_model, predict_smiles
from bbb_molgraph.explainability.attribution import DualChannelAttributionExtractor
from bbb_molgraph.explainability.visualizer import PharmacophoreVisualizer


def test_attribution_extractor():
    model = load_pretrained_model("default")
    extractor = DualChannelAttributionExtractor(model=model)
    smiles = "CCO"  # Ethanol (3 heavy atoms)

    explanation = extractor.explain_smiles(smiles)
    assert "probability" in explanation
    assert "atom_attentions" in explanation
    assert len(explanation["atom_attentions"]) == 3
    assert 0.0 <= explanation["probability"] <= 1.0


def test_attribution_extractor_single_channel_mpnn():
    from bbb_molgraph.models.mpnn import MPNNClassifier
    from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer

    featurizer = MolecularGraphFeaturizer()
    model = MPNNClassifier(
        atomic_dim=featurizer.atom_dim,
        bond_dim=featurizer.bond_dim,
        hidden_dim=32,
    )
    extractor = DualChannelAttributionExtractor(model=model)
    smiles = "CCO"

    explanation = extractor.explain_smiles(smiles)
    assert "probability" in explanation
    assert "atom_attentions" in explanation
    assert len(explanation["atom_attentions"]) == 3
    assert 0.0 <= explanation["probability"] <= 1.0


def test_pharmacophore_visualizer_svg(tmp_path):
    smiles = "CC(=O)O"  # Acetic acid (4 heavy atoms)
    weights = np.array([0.1, 0.4, 0.8, 0.2])

    visualizer = PharmacophoreVisualizer(colormap="Reds", image_size=(300, 300))
    svg_file = tmp_path / "test_molecule.svg"
    svg_content = visualizer.render_to_svg(smiles, weights, output_path=svg_file)

    assert svg_file.exists()
    assert "<svg" in svg_content
    assert "</svg>" in svg_content
