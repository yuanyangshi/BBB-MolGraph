# BBB-MolGraph Documentation

Welcome to **BBB-MolGraph**, a publication-grade Python deep learning framework designed for accurate and interpretable **Blood-Brain Barrier Permeability (BBBP)** prediction.

## Overview
BBB-MolGraph combines two complementary representations of small molecules:
1. **Topological Molecular Graphs (MPNN)**: Encodes 2D atomic node features, chemical bond orders, and message passing dynamics.
2. **Pretrained Sequence Transformers (MolFormer)**: Leverages large-scale chemical language models to extract global contextual embeddings.
3. **Cross-Modal Gated Fusion**: Adaptively integrates graph and sequence modalities for optimal generalization.

## Core Features
- **Strict Scientific Reproducibility**: Seed locking across CPU/CUDA/cuDNN and fixed 5-fold Bemis-Murcko scaffold splitting.
- **Top-Tier Journal Metrics**: Full reporting of ROC-AUC, PR-AUC, Balanced Accuracy, MCC, Sensitivity, Specificity, and 95% Bootstrap Confidence Intervals.
- **Explainability & Attribution**: Dual-channel attention mapping with publication-ready 2D pharmacophore structure heatmaps (SVG/PNG at 600 DPI).
- **Production-Ready Engineering**: Modular `src/` layout, CLI tools (`bbb-train`, `bbb-eval`, `bbb-predict`), unit tests (`pytest`), and continuous integration.
