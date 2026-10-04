"""
Configuration parser supporting YAML files, nested overrides, and path normalization.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_project_root() -> Path:
    """
    Dynamically locate the repository root without hardcoded machine paths.
    """
    # Assuming config_parser is at src/bbb_molgraph/utils/config_parser.py
    return Path(__file__).resolve().parents[3]


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dict2 into dict1.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file into a dictionary.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = get_project_root() / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Handle optional base config inheritance via '_base_' key
    if "_base_" in config:
        base_path = Path(config.pop("_base_"))
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base_config = load_yaml_config(base_path)
        config = deep_merge(base_config, config)

    return config


def parse_cli_overrides(override_args: list) -> Dict[str, Any]:
    """
    Parse a list of KEY=VALUE strings into a nested dictionary.
    Example: ['training.lr=0.001', 'model.dropout=0.3'] -> {'training': {'lr': 0.001}, 'model': {'dropout': 0.3}}
    """
    overrides: Dict[str, Any] = {}
    for arg in override_args:
        if "=" not in arg:
            continue
        key, val_str = arg.split("=", 1)
        # Attempt type casting
        try:
            val = yaml.safe_load(val_str)
        except Exception:
            val = val_str

        tokens = key.strip().split(".")
        d = overrides
        for token in tokens[:-1]:
            if token not in d or not isinstance(d[token], dict):
                d[token] = {}
            d = d[token]
        d[tokens[-1]] = val

    return overrides
