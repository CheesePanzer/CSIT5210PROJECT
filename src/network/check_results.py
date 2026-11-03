"""核对正式结果的覆盖、参数、划分、训练历史及 JSON/预测 CSV 的指标一致性。

不重新训练模型。运行：python -m src.network.check_results
"""

import argparse
import hashlib
import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets import load_dataset
from src.network.reproduce_split import INDEX_FIELDS, dataset_fingerprint, sample_counts
from src.split import get_inner_subsets, get_outer_folds


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("accuracy", "macro_f1", "g_mean", "ba")
RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]


def recompute_metrics(y_true, y_pred, k):
    """从混淆矩阵独立计算，避免只把保存过的分数再抄一遍。"""
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    correct = np.diag(matrix).astype(float)
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    recall = np.divide(correct, actual, out=np.zeros(k), where=actual != 0)
    f1 = np.divide(2 * correct, actual + predicted, out=np.zeros(k), where=(actual + predicted) != 0)
    return {"accuracy": float(correct.sum() / len(y_true)), "macro_f1": float(f1.mean()),
            "g_mean": 0.0 if np.any(recall == 0) else float(np.exp(np.log(recall).mean())),
            "ba": float(recall.mean()), "confusion_matrix": matrix.tolist()}


def verify_metrics(saved, actual, location):
    for metric in METRICS:
        if not math.isfinite(saved[metric]) or not math.isclose(saved[metric], actual[metric], abs_tol=1e-10):
            raise ValueError(f"{location}: {metric} does not match predictions")
    if saved["confusion_matrix"] != actual["confusion_matrix"]:
        raise ValueError(f"{location}: confusion matrix differs")


def audit(results_dir, data_dir):
    registry = pd.read_csv(ROOT / "data" / "dataset_registry.csv").set_index("dataset_id")
    expected = set(itertools.product(registry.index, ("ce", "ovr", "ovo"), ("accuracy", "macro_f1")))
    seen, issues, rows, fold_rows, manifests, cached, configs = set(), [], [], [], [], {}, {}
    loss_profiles = set()
    history_count, epoch_count = 0, 0
    paths = sorted(results_dir.glob("*.json"))
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            ds, loss, criterion = report["dataset"], report["loss"], report["criterion"]
            loss_profiles.add(report.get("loss_profile", report["implementation"]))
            key = (ds, loss, criterion)
            if key in seen or key not in expected:
                raise ValueError(f"Duplicate/unexpected experiment: {key}")
            seen.add(key)
            if not report["complete_five_fold_run"] or report["selected_folds"] != [1, 2, 3, 4, 5]:
                raise ValueError("Not a complete five-fold run")
            if (report["seed"], report["batch_size"], report["max_epochs"], report["patience"]) != (42, 8, 40, 10):
                raise ValueError("Training settings differ from the agreed full experiment")
            if report["optimizer"] != "torch.optim.Adam" or report["learning_rates"] != RATES:
                raise ValueError("Optimizer/learning-rate grid differs")
            if ds not in cached:
                X, y, meta = load_dataset(ds, str(data_dir))
                outer = get_outer_folds(X, y, metadata=meta, seed=42)
                partitions = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    for train_val, test in outer:
                        tr, va = get_inner_subsets(X, train_val, y, seed=42)[0]
                        partitions.append({"train_val": train_val, "test": test,
                                           "inner_train": train_val[tr], "inner_val": train_val[va]})
                cached[ds] = (y, meta, partitions, dataset_fingerprint(X, y, meta))
            y, meta, partitions, fingerprint = cached[ds]
            n, k = len(y), meta["n_classes"]
            if n != int(registry.loc[ds, "raw_sample_count"]) or k != int(registry.loc[ds, "final_class_count"]):
                raise ValueError("Current dataset differs from the registry")
            if report["class_labels"] != meta["class_labels"]:
                raise ValueError("Class encoding differs from the current data loader")
            if report["total_samples"] != n or report["evaluated_samples"] != n:
                raise ValueError("Wrong sample coverage")
            config = report.get("split_config")
            if config is not None:
                if config["dataset_sha256"] != fingerprint or config["seed"] != 42:
                    raise ValueError("Data fingerprint or split seed differs")
                if ds in configs and configs[ds] != config:
                    raise ValueError("Split configuration differs between losses/criteria")
                configs[ds] = config
            csv_path = path.with_name(path.stem + "_predictions.csv")
            frame = pd.read_csv(csv_path)
            if len(frame) != n or frame["sample_index"].duplicated().any():
                raise ValueError("Duplicate/missing prediction samples")
            for col in ("sample_index", "fold", "y_true", "y_pred"):
                if frame[col].isna().any() or not np.equal(frame[col], frame[col].astype(np.int64)).all():
                    raise ValueError(f"Invalid integer column: {col}")
            frame = frame.sort_values("sample_index")
            np.testing.assert_array_equal(frame["sample_index"], np.arange(n))
            np.testing.assert_array_equal(frame["y_true"], y)
            predictions = frame["y_pred"].to_numpy(dtype=np.int64)
            if np.any(predictions < 0) or np.any(predictions >= k):
                raise ValueError("Predicted class outside valid range")
            verify_metrics(report["combined_test_metrics"], recompute_metrics(y, predictions, k), "combined")
            folds = report["fold_results"]
            if sorted(f["fold"] for f in folds) != [1, 2, 3, 4, 5]:
                raise ValueError("Fold records are missing or duplicated")
            for fold in folds:
                number = fold["fold"]
                p = partitions[number - 1]
                counts = sample_counts(p["train_val"], p["test"], p["inner_train"], p["inner_val"])
                if "sample_counts" in fold and fold["sample_counts"] != counts:
                    raise ValueError(f"Fold {number}: partition counts differ")
                for partition, field in INDEX_FIELDS.items():
                    if field in fold:
                        np.testing.assert_array_equal(fold[field], p[partition])
                np.testing.assert_array_equal(frame.loc[frame["fold"] == number, "sample_index"], p["test"])
                verify_metrics(fold["test_metrics"], recompute_metrics(y[p["test"]], predictions[p["test"]], k), f"fold {number}")
                trials = fold["trials"]
                if [trial["lr"] for trial in trials] != RATES:
                    raise ValueError(f"Fold {number}: incomplete learning-rate search")
                for trial in trials:
                    history = trial["history"]
                    if not 1 <= len(history) <= 40 or [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
                        raise ValueError("Invalid epoch history")
                    values = np.asarray([[r["train_loss"], r["val_score"]] for r in history])
                    if not np.isfinite(values).all() or np.any(values[:, 1] < 0) or np.any(values[:, 1] > 1):
                        raise ValueError("Nonfinite loss or invalid validation score")
                    best_index = int(np.argmax(values[:, 1]))
                    # skorch 旧报告与 PyTorch 手动版使用不同字段名，含义相同。
                    best_epoch = trial.get("epochs", trial.get("best_epoch"))
                    best_score = trial.get("score", trial.get("val_score"))
                    if best_epoch != best_index + 1 or best_score != values[best_index, 1]:
                        raise ValueError("Recorded best validation epoch/score differs")
                    if len(history) < 40 and len(history) - best_epoch != 10:
                        raise ValueError("Unexpected early-stopping history")
                    history_count += 1
                    epoch_count += len(history)
                best = max(trials, key=lambda trial: trial.get("score", trial.get("val_score")))
                if fold["best_lr"] != best["lr"] or fold["refit_epochs"] != best.get("epochs", best.get("best_epoch")):
                    raise ValueError("Refit settings differ from selected trial")
                fold_rows.append({"dataset": ds, "loss": loss, "selection_criterion": criterion,
                                  "fold": number, "best_lr": fold["best_lr"], "refit_epochs": fold["refit_epochs"],
                                  **counts, **{metric: fold["test_metrics"][metric] for metric in METRICS}})
            rows.append({"dataset": ds, "selection_criterion": criterion, "loss": loss,
                         "imbalance_group": registry.loc[ds, "imbalance_group"], "n_samples": n, "n_classes": k,
                         **{metric: report["combined_test_metrics"][metric] for metric in METRICS},
                         "seed": report["seed"], "device": report["device"], "source_report": path.name})
            for source in (path, csv_path):
                manifests.append({"path": source.name, "bytes": source.stat().st_size,
                                  "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        except Exception as error:
            issues.append({"file": path.name, "error": f"{type(error).__name__}: {error}"})
    missing = sorted(expected - seen)
    if len(loss_profiles) > 1:
        issues.append({"file": "collection", "error": "Mixed loss implementations; keep different versions in separate result directories."})
    return {"source_directory": str(results_dir.resolve()), "expected_experiments": len(expected),
            "loss_profiles": sorted(loss_profiles),
            "found_json": len(paths), "found_predictions_csv": len(list(results_dir.glob("*_predictions.csv"))),
            "validated_experiments": len(rows), "validated_folds": len(fold_rows),
            "validated_lr_trials": history_count, "checked_history_epochs": epoch_count,
            "missing_experiments": missing, "issues": issues,
            "passed": not issues and not missing and len(rows) == len(expected),
            "metrics": rows, "fold_details": fold_rows, "source_manifest": manifests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "neural_network_skorch")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "nn_handoff" / "audit_data.json")
    args = parser.parse_args()
    report = audit(args.results, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("metrics", "fold_details", "source_manifest")},
                     ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
