# Quickstart Guide

Get up and running with BBB-MolGraph in minutes.

## 1. Installation
Install from source in editable mode:
```bash
git clone https://github.com/anonymous/BBB-MolGraph.git
cd BBB-MolGraph
pip install -e .
```

## 2. Five Lines of Python
Predict BBB permeability for any candidate molecule:
```python
from bbb_molgraph import load_pretrained_model, predict_smiles

model = load_pretrained_model("default")
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

prob, explanation = predict_smiles(smiles, model=model, explain=True)
print(f"BBB Permeability Probability: {prob:.4f}")
print(f"Number of atoms analyzed: {explanation['num_atoms']}")
```

## 3. Command-Line Interface (CLI)
### Training on Bemis-Murcko Scaffolds
```bash
bbb-train --config configs/experiment/scaffold_5fold_cv.yaml
```

### Batch Prediction
```bash
bbb-predict --input-csv data/custom_molecules.csv --output-csv results/predictions.csv
```

### Model Evaluation
```bash
bbb-eval --model-path checkpoints/best.pt --data-csv data/raw/BBBP_combined.csv
```
