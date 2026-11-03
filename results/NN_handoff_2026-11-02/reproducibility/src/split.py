import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_outer_folds(
    X,
    y: np.ndarray,
    metadata: Optional[dict] = None,
    n_splits: int = 5,
    seed: int = 42
):
    # If predefined folds are available in the metadata, they are returned; otherwise, new stratified folds are generated using StratifiedKFold.

    if metadata is not None and metadata.get("predefined_folds") is not None:
        return metadata["predefined_folds"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)]


def get_inner_subsets(
    X,
    train_val_idx: np.ndarray,
    y: np.ndarray,
    n_subsets: int = 5,
    seed: int = 42
):
    # Generates inner subsets for cross-validation within the training/validation set, using StratifiedKFold to maintain class distribution.

    y_train_val = y[train_val_idx]
    X_train_val = X.iloc[train_val_idx]
    skf = StratifiedKFold(n_splits=n_subsets, shuffle=True, random_state=seed)
    return [(inner_tr, inner_val) for inner_tr, inner_val in skf.split(X_train_val, y_train_val)]


def save_splits_to_disk(dataset_id: str, outer_folds: List[Tuple[np.ndarray, np.ndarray]], out_dir: str = "data/splits"):
# Saves the outer folds of a dataset to disk in JSON format, allowing for reproducibility and easy access to the train/test splits.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    serializable = [
        {"fold": i + 1, "train_idx": tr.tolist(), "test_idx": te.tolist()}
        for i, (tr, te) in enumerate(outer_folds)
    ]
    out_path = Path(out_dir) / f"{dataset_id.upper()}_outer_splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)