"""Build the two paper-style IR-group tables for the final neural results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASET_GROUPS = {
    "AUT": "high", "BAL": "medium", "CAR": "high", "CLE": "high",
    "DER": "medium", "ECO": "high", "FLA": "medium", "GLA": "medium",
    "HAY": "low", "HCV": "high", "LYM": "high", "NEW": "medium",
    "SEG": "low", "SHU": "high", "THY": "high", "VEH": "low",
    "ZOO": "high",
}
GROUP_ORDER = ["low", "medium", "high"]
METRICS = ["accuracy", "macro_f1", "g_mean", "ba"]
METHODS = ["CE", "OVR", "OVO-P", "OVO-A"]


def load_reports(folder: Path, implementation: str, losses: set[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("loss") not in losses:
            continue
        metrics = report["combined_test_metrics"]
        rows.append({
            "implementation": implementation,
            "dataset": report["dataset"],
            "criterion": report["criterion"],
            "loss": report["loss"],
            **{metric: metrics[metric] for metric in METRICS},
            "source_report": path.name,
        })
    return pd.DataFrame(rows)


def validate_complete(frame: pd.DataFrame, implementation: str, expected_losses: set[str]) -> None:
    expected = {(dataset, criterion, loss)
                for dataset in DATASET_GROUPS
                for criterion in ("accuracy", "macro_f1")
                for loss in expected_losses}
    found = set(zip(frame["dataset"], frame["criterion"], frame["loss"]))
    missing = sorted(expected - found)
    duplicates = frame.duplicated(["dataset", "criterion", "loss"]).sum()
    if missing or duplicates or len(frame) != len(expected):
        raise ValueError(
            f"{implementation} results incomplete: rows={len(frame)}, "
            f"expected={len(expected)}, duplicates={duplicates}, missing={missing[:10]}"
        )


def combine_results(paper: pd.DataFrame, author: pd.DataFrame) -> pd.DataFrame:
    paper = paper.copy()
    author = author.copy()

    # CE and OVR/OVO-P come from the paper-formula protocol. OVO-A comes
    # from the author-code protocol so the table compares one CE baseline,
    # OVR, and both OVO implementations.
    paper["method"] = paper["loss"].map({"ce": "CE", "ovr": "OVR", "ovo": "OVO-P"})
    author["method"] = author["loss"].map({"ovo": "OVO-A"})
    paper = paper[paper["method"].notna()]
    author = author[author["method"].notna()]

    combined = pd.concat([paper, author], ignore_index=True)
    combined["imbalance_group"] = combined["dataset"].map(DATASET_GROUPS)
    if combined["imbalance_group"].isna().any():
        raise ValueError("At least one result dataset has no imbalance-group mapping.")
    return combined


def make_table(combined: pd.DataFrame, criterion: str) -> pd.DataFrame:
    subset = combined[combined["criterion"] == criterion]
    grouped = (
        subset.groupby(["imbalance_group", "method"], sort=False)[METRICS]
        .mean()
        .reindex(pd.MultiIndex.from_product(
            [GROUP_ORDER, METHODS], names=["imbalance_group", "method"]
        ))
    )

    columns = []
    values = []
    for group in GROUP_ORDER:
        row = {"Imbalance group": group}
        for metric in METRICS:
            for method in METHODS:
                row[f"{metric}_{method}"] = grouped.loc[(group, method), metric]
        values.append(row)

    flat = pd.DataFrame(values)
    flat.columns = pd.MultiIndex.from_tuples(
        [("", "Imbalance group")]
        + [(metric.upper() if metric != "macro_f1" else "Macro-F1", method)
           for metric in METRICS for method in METHODS]
    )
    return flat


def make_per_dataset_table(combined: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """Create one row per dataset plus an equal-weight average row."""
    subset = combined[combined["criterion"] == criterion]
    grouped = subset.groupby(["dataset", "method"], sort=False)[METRICS].mean()
    rows = []

    for dataset in sorted(DATASET_GROUPS):
        row = {"Dataset": dataset}
        for metric in METRICS:
            for method in METHODS:
                row[f"{metric}_{method}"] = grouped.loc[(dataset, method), metric]
        rows.append(row)

    average = {"Dataset": "Average"}
    for metric in METRICS:
        for method in METHODS:
            average[f"{metric}_{method}"] = sum(
                row[f"{metric}_{method}"] for row in rows
            ) / len(rows)
    rows.append(average)

    table = pd.DataFrame(rows)
    table.columns = pd.MultiIndex.from_tuples(
        [("", "Dataset")]
        + [(metric.upper() if metric != "macro_f1" else "Macro-F1", method)
           for metric in METRICS for method in METHODS]
    )
    return table


def make_average_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Create a two-row overall table, one row per selection criterion."""
    grouped = combined.groupby(["criterion", "method"], sort=False)[METRICS].mean()
    rows = []
    for criterion in ("accuracy", "macro_f1"):
        row = {"Selection criterion": criterion}
        for metric in METRICS:
            for method in METHODS:
                row[f"{metric}_{method}"] = grouped.loc[(criterion, method), metric]
        rows.append(row)

    table = pd.DataFrame(rows)
    table.columns = pd.MultiIndex.from_tuples(
        [("", "Selection criterion")]
        + [(metric.upper() if metric != "macro_f1" else "Macro-F1", method)
           for metric in METRICS for method in METHODS]
    )
    return table


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    display = table.copy()
    display.columns = [
        "Imbalance group" if not second else f"{first} ({second})"
        for first, second in display.columns
    ]
    path.write_text(display.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")


def flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    flattened = table.copy()
    flattened.columns = [
        "Imbalance group" if not second else f"{first}_{second}"
        for first, second in flattened.columns
    ]
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--author-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paper = load_reports(args.paper_dir, "paper_formula", {"ce", "ovr", "ovo"})
    author = load_reports(args.author_dir, "author_code", {"ovo"})
    validate_complete(paper, "paper_formula", {"ce", "ovr", "ovo"})
    validate_complete(author, "author_code", {"ovo"})

    combined = combine_results(paper, author)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "combined_dataset_results.csv", index=False)

    accuracy_table = make_table(combined, "accuracy")
    macro_f1_table = make_table(combined, "macro_f1")
    accuracy_per_dataset = make_per_dataset_table(combined, "accuracy")
    macro_f1_per_dataset = make_per_dataset_table(combined, "macro_f1")
    average_table = make_average_table(combined)

    accuracy_table.to_csv(args.output_dir / "accuracy_selection_by_imbalance.csv", index=False)
    macro_f1_table.to_csv(args.output_dir / "macro_f1_selection_by_imbalance.csv", index=False)
    accuracy_per_dataset.to_csv(args.output_dir / "neural_paper_style_results_perdataset_accuracy.csv", index=False)
    macro_f1_per_dataset.to_csv(args.output_dir / "neural_paper_style_results_perdataset_macro_f1.csv", index=False)
    average_table.to_csv(args.output_dir / "neural_paper_style_results_average.csv", index=False)
    write_markdown(accuracy_table, args.output_dir / "accuracy_selection_by_imbalance.md")
    write_markdown(macro_f1_table, args.output_dir / "macro_f1_selection_by_imbalance.md")
    write_markdown(accuracy_per_dataset, args.output_dir / "neural_paper_style_results_perdataset_accuracy.md")
    write_markdown(macro_f1_per_dataset, args.output_dir / "neural_paper_style_results_perdataset_macro_f1.md")
    write_markdown(average_table, args.output_dir / "neural_paper_style_results_average.md")

    with pd.ExcelWriter(args.output_dir / "neural_final_results.xlsx") as writer:
        flatten_columns(accuracy_table).to_excel(writer, sheet_name="Accuracy selection", index=False)
        flatten_columns(macro_f1_table).to_excel(writer, sheet_name="Macro-F1 selection", index=False)
        flatten_columns(accuracy_per_dataset).to_excel(writer, sheet_name="Per-dataset Accuracy", index=False)
        flatten_columns(macro_f1_per_dataset).to_excel(writer, sheet_name="Per-dataset Macro-F1", index=False)
        flatten_columns(average_table).to_excel(writer, sheet_name="Average", index=False)
        combined.to_excel(writer, sheet_name="Raw Dataset Results", index=False)

    print(f"Paper reports: {len(paper)}")
    print(f"Author OVO reports: {len(author)}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
