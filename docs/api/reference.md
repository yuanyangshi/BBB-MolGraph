# API Reference

Detailed technical reference for `bbb_molgraph`.

## Data Pipeline
### `bbb_molgraph.data.MolecularGraphFeaturizer`
Converts SMILES strings into topological graph tensors.
- `smiles_to_graph(smiles: str) -> Dict[str, torch.Tensor]`
- `clear_cache() -> None`

### `bbb_molgraph.data.ScaffoldSplitter`
Implements Bemis-Murcko scaffold partitioning to prevent train/test chemical overlap.
- `k_fold_scaffold_cv(smiles_list, k=5, seed=42) -> List[Tuple[List[int], List[int]]]`
- `split_train_val_test(smiles_list, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42)`

## Models
### `bbb_molgraph.models.DualChannelBBBPredictor`
Dual-channel multimodal neural network combining MPNN and MolFormer.
- Parameters:
  - `atomic_dim: int`
  - `bond_dim: int`
  - `mpnn_hidden_dim: int`
  - `molformer_model_name: str`
  - `fusion_mode: str` ('gated', 'concat')

### `bbb_molgraph.models.MPNNClassifier`
Single-modality message passing graph neural network baseline.

## Training & Evaluation
### `bbb_molgraph.training.Trainer`
Scientific training engine equipped with AMP mixed-precision, gradient clipping, and callbacks.
- `fit(train_loader, val_loader, max_epochs) -> Dict[str, List[float]]`
- `evaluate(val_loader) -> Dict[str, float]`

### `bbb_molgraph.evaluation.compute_classification_metrics`
Computes ROC-AUC, PR-AUC, MCC, Balanced Accuracy, Sensitivity, Specificity, F1, and Brier Score.

### `bbb_molgraph.evaluation.bootstrap_confidence_interval`
Computes 95% Bootstrap Confidence Intervals over 1,000 resamples.

## Explainability
### `bbb_molgraph.explainability.PharmacophoreVisualizer`
Renders 2D molecular structures with atom-level attention heatmap coloring.
- `render_to_svg(smiles, atom_weights, output_path) -> str`
- `render_to_png(smiles, atom_weights, output_path) -> None`
