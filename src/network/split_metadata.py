"""Shared split metadata helpers used by training and audit tools."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn


INDEX_FIELDS = {
    "train_val": "train_val_indices",
    "test": "test_indices",
    "inner_train": "inner_train_indices",
    "inner_val": "inner_val_indices",
}


def dataset_fingerprint(X, y, metadata):
    """Hash the loaded data, ordering, labels, and predefined fold boundaries."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy(dtype="<u8").tobytes())
    digest.update(np.asarray(y, dtype="<i8").tobytes())
    digest.update(json.dumps(
        {"columns": list(X.columns), "labels": metadata["class_labels"]},
        ensure_ascii=False,
    ).encode("utf-8"))

    for _, test_idx in metadata.get("predefined_folds") or []:
        digest.update(np.asarray([len(test_idx)], dtype="<i8").tobytes())
        digest.update(np.asarray(test_idx, dtype="<i8").tobytes())

    return digest.hexdigest()


def make_split_config(X, y, metadata, data_dir, seed):
    """Build the split and data-version metadata stored in experiment reports."""
    return {
        "outer_method": "keel_predefined" if metadata.get("has_predefined_folds") else "stratified_kfold",
        "outer_n_splits": 5,
        "inner_n_splits": 5,
        "inner_fold": 1,
        "shuffle_generated_folds": True,
        "seed": int(seed),
        "data_dir": str(Path(data_dir).resolve()),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "dataset_sha256": dataset_fingerprint(X, y, metadata),
    }


def sample_counts(train_val_idx, test_idx, train_idx, val_idx):
    """Return partition sizes for an experiment report."""
    return {
        "train_val": len(train_val_idx),
        "inner_train": len(train_idx),
        "inner_val": len(val_idx),
        "test": len(test_idx),
    }
