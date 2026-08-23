"""Load config/default.yaml and expose repo-root-relative paths.

Every script in src/ calls load_config() instead of hardcoding paths or
hyperparameters, so a single YAML file is the source of truth for a
reproducible `make all` run (see SPEC.md Q1.5).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a paths.<key> entry from config relative to the repo root."""
    rel = cfg["paths"][key]
    return REPO_ROOT / rel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# (dataset, split) pairs used by tests and eval scripts to enumerate all splits.
SPLITS = [
    ("mind", "train"),
    ("mind", "val"),
    ("mind", "test"),
    ("ebnerd", "train"),
    ("ebnerd", "val"),
    ("ebnerd", "test"),
]
