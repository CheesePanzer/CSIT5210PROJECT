#  """按结果 JSON 复现数据划分，需要时才导出完整样本归属。
#
# 运行：python -m src.network.tools.reproduce_split --report <结果.json> --fold 1
# 这里只重建划分，不训练模型，也不修改原来的结果文件。
# """

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.datasets import load_dataset
from src.split import get_inner_subsets, get_outer_folds
from src.network.split_metadata import (
    INDEX_FIELDS,
    dataset_fingerprint,
)


ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "src" / "network"
def reconstruct_split(report_path, fold, data_dir=None):
    """返回分区索引、整数标签、元数据；同时兼容以前保存索引的旧报告。"""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    recorded_fold = next((item for item in report["fold_results"] if item["fold"] == fold), None)
    if recorded_fold is None:
        raise ValueError(f"Fold {fold} was not run in this report.")
    config = report.get("split_config", {})
    # 换电脑后可用 --data-dir 指向同一版本数据，不要求目录绝对路径相同。
    data_dir = data_dir or config.get("data_dir") or ROOT / "data" / "raw"
    X, y, meta = load_dataset(report["dataset"], data_dir=str(data_dir))
    if config:
        if config["sklearn_version"] != sklearn.__version__ or config["pandas_version"] != pd.__version__:
            raise ValueError("Use the sklearn/pandas versions recorded in split_config to reproduce this split.")
        if dataset_fingerprint(X, y, meta) != config["dataset_sha256"]:
            raise ValueError("Dataset contents, row order or official folds differ from the saved report.")
        expected_method = "keel_predefined" if meta.get("has_predefined_folds") else "stratified_kfold"
        if config["outer_method"] != expected_method:
            raise ValueError("Outer split method differs from the saved report.")

    # 所有划分使用报告顶层的原始 seed；不要使用每折训练时的 seed+fold。
    seed = config.get("seed", report["seed"])
    outer = get_outer_folds(X, y, metadata=meta, seed=seed,
                            n_splits=config.get("outer_n_splits", 5))
    train_val, test = outer[fold - 1]
    inner = get_inner_subsets(X, train_val, y, seed=seed,
                              n_subsets=config.get("inner_n_splits", 5))
    train_rel, val_rel = inner[config.get("inner_fold", 1) - 1]
    partitions = {"train_val": train_val, "test": test,
                  "inner_train": train_val[train_rel], "inner_val": train_val[val_rel]}
    counts = {name: len(indices) for name, indices in partitions.items()}
    if "sample_counts" in recorded_fold and counts != recorded_fold["sample_counts"]:
        raise ValueError("Reconstructed sample counts differ from the report.")
    # 旧报告没有数据指纹，但有完整索引，可以直接验证重建的结果是否一致。
    for name, field in INDEX_FIELDS.items():
        if field in recorded_fold and not np.array_equal(partitions[name], recorded_fold[field]):
            raise ValueError(f"Reconstructed {name} indices differ from this legacy report.")
    return partitions, y, meta


def export_membership(partitions, y, metadata, output):
    """一行对应加载后的一条样本；outer/inner 两列说明它属于哪个分区。"""
    outer = np.full(len(y), "train_val", dtype=object)
    inner = np.full(len(y), "not_used", dtype=object)
    outer[partitions["test"]] = "test"
    inner[partitions["inner_train"]] = "train"
    inner[partitions["inner_val"]] = "validation"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_index": np.arange(len(y)), "y": y,
        "class_label": np.asarray(metadata["class_labels"])[y],
        "outer_partition": outer, "inner_partition": inner,
    }).to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--data-dir", type=Path, help="Override the raw dataset location")
    parser.add_argument("--output", type=Path, help="Optional membership CSV path")
    args = parser.parse_args()
    try:
        partitions, y, meta = reconstruct_split(args.report, args.fold, args.data_dir)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    output = args.output or NETWORK_DIR / "results" / "splits" / f"{args.report.stem}_fold{args.fold}_membership.csv"
    export_membership(partitions, y, meta, output)
    print("Sample counts:", {name: len(values) for name, values in partitions.items()})
    print("sample_index is the zero-based row position returned by load_dataset, not a raw-file line number.")
    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
