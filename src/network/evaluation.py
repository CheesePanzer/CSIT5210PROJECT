"""Shared metric logic for validation selection and final prediction analysis."""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import gmean
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


class PaperEvaluation:
    def __init__(self, num_classes: int):
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.labels = list(range(num_classes))

    def score(self, y_true, y_pred, criterion: str) -> float:
        if criterion == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if criterion == "macro_f1":
            return float(f1_score(y_true, y_pred, labels=self.labels, average="macro", zero_division=0))
        raise ValueError(f"Unsupported selection criterion: {criterion}")

    def evaluate(self, y_true, y_pred) -> dict:
        recalls = recall_score(y_true, y_pred, labels=self.labels, average=None, zero_division=0)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=self.labels, average="macro", zero_division=0)),
            "g_mean": float(gmean(recalls)),
            "ba": float(recall_score(y_true, y_pred, labels=self.labels, average="macro", zero_division=0)),
            "per_class_recall": recalls.tolist(),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=self.labels).tolist(),
        }


def evaluate_prediction_csv(path: str | Path, num_classes: int) -> dict:
    frame = pd.read_csv(path)
    missing = {"y_true", "y_pred"} - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    return PaperEvaluation(num_classes).evaluate(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())


def save_evaluation(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
