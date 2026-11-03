"""检查精简报告仍能重建相同划分，并能发现数据/版本变化。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.network.reproduce_split import (
    INDEX_FIELDS, export_membership, make_split_config, reconstruct_split, sample_counts,
)
from src.split import get_inner_subsets, get_outer_folds


class SplitReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.X = pd.DataFrame({"f": np.arange(60, dtype=float)})
        self.y = np.tile(np.arange(3), 20)
        self.meta = {"class_labels": ["A", "B", "C"],
                     "has_predefined_folds": False, "predefined_folds": None}
        train_val, test = get_outer_folds(self.X, self.y, self.meta, seed=42)[0]
        train_rel, val_rel = get_inner_subsets(self.X, train_val, self.y, seed=42)[0]
        self.expected = {"train_val": train_val, "test": test,
                         "inner_train": train_val[train_rel], "inner_val": train_val[val_rel]}
        self.report = {
            "dataset": "SYN", "seed": 42,
            "split_config": make_split_config(self.X, self.y, self.meta, self.root, 42),
            "fold_results": [{"fold": 1, "seed": 43, "sample_counts": sample_counts(
                train_val, test, train_val[train_rel], train_val[val_rel]
            )}],
        }
        self.report_path = self.root / "report.json"

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def reconstruct(self, X=None, y=None):
        self.write_report()
        with patch("src.network.reproduce_split.load_dataset", return_value=(
            self.X if X is None else X, self.y if y is None else y, self.meta
        )):
            return reconstruct_split(self.report_path, 1)

    def test_round_trip_uses_split_seed_not_training_seed(self):
        partitions, _, _ = self.reconstruct()
        for key, expected in self.expected.items():
            np.testing.assert_array_equal(partitions[key], expected)
        self.assertFalse(set(partitions["test"]) & set(partitions["train_val"]))

    def test_changed_row_order_is_detected(self):
        with self.assertRaisesRegex(ValueError, "contents, row order"):
            self.reconstruct(self.X.iloc[::-1].reset_index(drop=True), self.y[::-1])

    def test_changed_library_version_is_detected(self):
        self.report["split_config"]["sklearn_version"] = "different-version"
        with self.assertRaisesRegex(ValueError, "versions"):
            self.reconstruct()

    def test_legacy_indices_verify_exact_membership(self):
        self.report.pop("split_config")
        for partition, field in INDEX_FIELDS.items():
            self.report["fold_results"][0][field] = self.expected[partition].tolist()
        self.reconstruct()
        self.report["seed"] = 43
        with self.assertRaisesRegex(ValueError, "legacy report"):
            self.reconstruct()

    def test_exported_csv_preserves_sample_membership(self):
        partitions, y, meta = self.reconstruct()
        output = self.root / "membership.csv"
        export_membership(partitions, y, meta, output)
        frame = pd.read_csv(output)
        np.testing.assert_array_equal(frame["sample_index"], np.arange(len(y)))
        for partition, column, value in (("test", "outer_partition", "test"),
                                         ("inner_train", "inner_partition", "train"),
                                         ("inner_val", "inner_partition", "validation")):
            np.testing.assert_array_equal(frame.loc[frame[column] == value, "sample_index"], partitions[partition])

    def test_wrong_saved_counts_are_detected(self):
        self.report["fold_results"][0]["sample_counts"]["test"] += 1
        with self.assertRaisesRegex(ValueError, "sample counts"):
            self.reconstruct()


if __name__ == "__main__":
    unittest.main()
