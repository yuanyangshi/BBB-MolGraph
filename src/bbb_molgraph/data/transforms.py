"""
Graph transforms, node feature standardizations, and chemical data augmentations.
"""

from typing import Dict
import torch


class NodeFeatureStandardizer:
    """
    Standardizes continuous node features using mean and standard deviation.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = torch.clamp(std, min=1e-6)

    def __call__(self, graph_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = graph_dict["node_features"]
        graph_dict["node_features"] = (features - self.mean) / self.std
        return graph_dict


class RandomNodeDropout:
    """
    Randomly drops node features with probability p during training as an augmentation.
    """

    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def __call__(self, graph_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = graph_dict["node_features"]
        mask = torch.rand_like(features) > self.p
        graph_dict["node_features"] = features * mask.float()
        return graph_dict
