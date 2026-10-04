"""
Classical machine learning and graph neural network baseline implementations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F


def smiles_to_morgan_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    Generate Morgan fingerprint (ECFP) bit vectors for a list of SMILES.
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.float32))
        else:
            fps.append(np.zeros(n_bits, dtype=np.float32))
    return np.stack(fps, axis=0)


class TabularMLBaseline:
    """
    Wrapper for Scikit-Learn classifiers evaluating Morgan fingerprints.
    """

    def __init__(self, model_type: str = "rf", random_state: int = 42, **kwargs) -> None:
        self.model_type = model_type
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "svm":
            self.model = SVC(
                C=kwargs.get("C", 1.0),
                probability=True,
                random_state=random_state,
            )
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=kwargs.get("max_iter", 1000),
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown tabular model type: {model_type}")

    def fit(self, smiles_train: List[str], y_train: np.ndarray) -> None:
        x_train = smiles_to_morgan_fingerprints(smiles_train)
        self.model.fit(x_train, y_train)

    def predict_proba(self, smiles_test: List[str]) -> np.ndarray:
        x_test = smiles_to_morgan_fingerprints(smiles_test)
        return self.model.predict_proba(x_test)[:, 1]


class SimpleGCNLayer(nn.Module):
    """Basic Graph Convolutional Network layer with self-loops and degree normalization."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        target, source = edge_index[:, 0], edge_index[:, 1]
        
        # Simple degree normalization
        deg = torch.zeros(num_nodes, device=x.device).scatter_add_(
            0, target, torch.ones_like(target, dtype=torch.float32)
        ).clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)

        norm = deg_inv_sqrt[target] * deg_inv_sqrt[source]
        h = self.linear(x)
        
        # Message passing
        msg = h[source] * norm.unsqueeze(-1)
        out = torch.zeros_like(h).scatter_add_(0, target.unsqueeze(-1).expand(-1, h.size(1)), msg)
        return out + self.bias


class GCNBaseline(nn.Module):
    """Standard 3-layer GCN baseline for molecular classification."""

    def __init__(self, node_dim: int = 30, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.gcn1 = SimpleGCNLayer(node_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch_graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch_graph["node_features"]
        edge_index = batch_graph["connectivity_indices"]
        batch_assignment = batch_graph["batch_assignment"]

        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index))

        # Global mean pooling
        batch_size = int(batch_assignment.max().item()) + 1
        pooled = torch.zeros(batch_size, x.size(1), device=x.device)
        counts = torch.zeros(batch_size, 1, device=x.device)
        pooled.scatter_add_(0, batch_assignment.unsqueeze(-1).expand(-1, x.size(1)), x)
        counts.scatter_add_(0, batch_assignment.unsqueeze(-1), torch.ones(x.size(0), 1, device=x.device))
        graph_emb = pooled / counts.clamp(min=1.0)

        return self.classifier(graph_emb)
