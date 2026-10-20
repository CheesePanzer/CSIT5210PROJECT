from datetime import date
from pathlib import Path
import pandas as pd

from src.datasets import load_dataset
from src.preprocessing import FoldPreprocessor
from src.split import get_outer_folds, save_splits_to_disk


def get_available_datasets(data_dir: str = "data/raw") -> list[str]:
    """
    Automatically discovers dataset IDs from files or folders inside data_dir.
    """
    raw_path = Path(data_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Data directory not found: {raw_path.resolve()}")

    dataset_ids = []
    for item in sorted(raw_path.iterdir()):
        if item.name.startswith(".") or "desktop" in item.name.lower():
            continue
        # If datasets are stored in folders, use folder name; if files (.csv/.dat), use file stem
        ds_id = item.name if item.is_dir() else item.stem
        if ds_id not in dataset_ids:
            dataset_ids.append(ds_id)

    return dataset_ids


def generate_dataset_registry(
    data_dir: str = "data/raw",
    registry_path: str = "data/dataset_registry.csv",
) -> pd.DataFrame:
    """
    Loads all local datasets, generates and saves the outer CV splits,
    verifies preprocessing dimensions, and saves a summary registry CSV.
    """
    rows = []
    dataset_ids = get_available_datasets(data_dir=data_dir)

    for ds_id in dataset_ids:
        X, y, meta = load_dataset(ds_id, data_dir=data_dir)
        outer_folds = get_outer_folds(X, y, metadata=meta)
        save_splits_to_disk(ds_id, outer_folds)

        # Check final feature dimension after One-Hot Encoding on the first fold
        tr_idx, _ = outer_folds[0]
        categorical_cols = meta.get("categorical_cols", [])
        prep = FoldPreprocessor(categorical_cols=categorical_cols)
        X_tr_proc = prep.fit_transform(X.iloc[tr_idx])

        rows.append({
            "dataset_id": ds_id,
            "download_date": str(date.today()),
            "raw_sample_count": meta.get("n_samples", len(X)),
            "raw_feature_count": meta.get("n_features_raw", X.shape[1]),
            "final_feature_count": int(X_tr_proc.shape[1]),
            "final_class_count": meta.get("n_classes", int( pd.Series(y).nunique() )),
            "raw_class_counts": str(meta.get("class_counts", pd.Series(y).value_counts().to_dict())),
            "computed_ir": meta.get("ir_computed", None),
            "imbalance_group": meta.get("imbalance_group", None),
            "categorical_columns": str(categorical_cols),
            "has_predefined_keel_folds": meta.get("has_predefined_folds", False),
        })

    df_reg = pd.DataFrame(rows)
    Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
    df_reg.to_csv(registry_path, index=False)
    return df_reg


if __name__ == "__main__":
    df = generate_dataset_registry()
    print(df.to_string(index=False))