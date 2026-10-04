"""
Unit tests for neural network architectures, forward passes, and gradients.
"""

import pytest
import torch
from bbb_molgraph.data.dataset import collate_molecular_graphs
from bbb_molgraph.data.featurizer import MolecularGraphFeaturizer
from bbb_molgraph.models.fusion import DualChannelBBBPredictor
from bbb_molgraph.models.molformer import MolFormerEncoder
from bbb_molgraph.models.mpnn import MPNNClassifier, MPNNEncoder


@pytest.fixture
def sample_batch():
    featurizer = MolecularGraphFeaturizer()
    smiles = ["CC(=O)O", "c1ccccc1"]
    items = []
    for s in smiles:
        g = featurizer.smiles_to_graph(s)
        items.append({
            "smiles": s,
            "node_features": g["node_features"],
            "edge_features": g["edge_features"],
            "connectivity_indices": g["connectivity_indices"],
            "num_nodes": g["num_nodes"],
            "label": 1.0,
        })
    return collate_molecular_graphs(items)


def test_mpnn_encoder_forward(sample_batch):
    atom_dim = sample_batch["node_features"].shape[1]
    bond_dim = sample_batch["edge_features"].shape[1]
    hidden_dim = 32

    encoder = MPNNEncoder(
        atomic_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=hidden_dim,
        propagation_steps=2,
    )
    embeddings, atom_attns = encoder(sample_batch)

    assert embeddings.shape == (2, hidden_dim)
    total_atoms = sample_batch["node_features"].shape[0]
    assert atom_attns.shape == (total_atoms,)


def test_mpnn_classifier_gradients(sample_batch):
    atom_dim = sample_batch["node_features"].shape[1]
    bond_dim = sample_batch["edge_features"].shape[1]

    model = MPNNClassifier(
        atomic_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=32,
    )
    logits, _ = model(sample_batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, sample_batch["labels"])
    loss.backward()

    # Verify gradients flow into model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"


def test_dual_channel_predictor(sample_batch):
    atom_dim = sample_batch["node_features"].shape[1]
    bond_dim = sample_batch["edge_features"].shape[1]

    model = DualChannelBBBPredictor(
        atomic_dim=atom_dim,
        bond_dim=bond_dim,
        mpnn_hidden_dim=32,
        molformer_out_dim=32,
        fusion_dim=64,
        fusion_mode="gated",
        use_fallback_molformer=True,
    )

    batch_size = sample_batch["batch_size"]
    dummy_input_ids = torch.randint(1, 100, (batch_size, 16))
    dummy_mask = torch.ones((batch_size, 16), dtype=torch.long)

    logits, attributions = model(sample_batch, dummy_input_ids, dummy_mask)
    assert logits.shape == (batch_size, 1)
    assert "graph_attentions" in attributions
    assert "seq_attentions" in attributions


def test_mpnn_classifier_from_pretrained_shape_inference(tmp_path, sample_batch):
    atom_dim = sample_batch["node_features"].shape[1]
    bond_dim = sample_batch["edge_features"].shape[1]
    custom_hidden = 48

    original = MPNNClassifier(
        atomic_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=custom_hidden,
    )
    ckpt_path = tmp_path / "mpnn_custom.pt"
    torch.save({"state_dict": original.state_dict()}, ckpt_path)

    # Load with automatic dimension inference
    loaded = MPNNClassifier.from_pretrained(ckpt_path)
    assert loaded.encoder.hidden_dim == custom_hidden
    assert loaded.encoder.atomic_dim == atom_dim
    assert loaded.encoder.bond_dim == bond_dim

    # Verify predictions
    with torch.no_grad():
        logits, _ = loaded(sample_batch)
    assert logits.shape == (sample_batch["batch_size"], 1)


def test_load_model_from_checkpoint_dispatch(tmp_path):
    from bbb_molgraph.models import load_model_from_checkpoint

    # 1. MPNN checkpoint
    mpnn = MPNNClassifier(atomic_dim=29, bond_dim=7, hidden_dim=32)
    mpnn_path = tmp_path / "mpnn.pt"
    torch.save({"state_dict": mpnn.state_dict()}, mpnn_path)
    loaded_mpnn = load_model_from_checkpoint(mpnn_path)
    assert isinstance(loaded_mpnn, MPNNClassifier)
    assert loaded_mpnn.encoder.hidden_dim == 32

    # 2. DualChannel checkpoint with fallback
    dual = DualChannelBBBPredictor(
        atomic_dim=29,
        bond_dim=7,
        mpnn_hidden_dim=32,
        molformer_out_dim=32,
        fusion_dim=64,
        use_fallback_molformer=True,
    )
    dual_path = tmp_path / "dual.pt"
    torch.save({"state_dict": dual.state_dict()}, dual_path)
    loaded_dual = load_model_from_checkpoint(dual_path)
    assert isinstance(loaded_dual, DualChannelBBBPredictor)
    assert loaded_dual.graph_encoder.hidden_dim == 32
    assert loaded_dual.seq_encoder.out_dim == 32

