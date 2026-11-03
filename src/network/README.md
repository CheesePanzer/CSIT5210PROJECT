# Neural Network Module

This directory contains the neural-network reproduction pipeline for the OVO/OVR imbalanced multi-class classification project.

## Directory Structure

```text
src/network/
├── network_train_torch_manual.py   # Main training entry point
├── paper_losses.py                 # CE, OVR, and OVO losses
├── split_metadata.py               # Shared split metadata helpers
├── tools/
│   ├── check_results.py            # Result-audit tool
│   └── reproduce_split.py          # Split inspection/export tool
└── archive/                        # Previous implementations; not for final runs
```

The data-loading, preprocessing, and fold-generation code is located one level above this directory:

```text
src/
├── datasets.py
├── preprocessing.py
├── split.py
└── dataset_registry.py
```

The network module reuses these shared interfaces instead of downloading or preprocessing datasets independently.

## Main Training Script

Use `network_train_torch_manual.py` for new experiments:

```powershell
python -m src.network.network_train_torch_manual --datasets CAR --device cpu
```

For CUDA, use:

```powershell
python -m src.network.network_train_torch_manual --datasets CAR --device cuda
```

The script supports:

- All available datasets with `--datasets ALL`;
- Cross-Entropy, OVR, and OVO losses;
- Accuracy and Macro-F1 hyperparameter-selection criteria;
- Eight candidate learning rates from the paper;
- Outer five-fold evaluation;
- Inner validation;
- Early stopping;
- CPU or CUDA execution;
- JSON experiment reports and per-sample prediction CSV files.

## Quick Smoke Test

Before running a full experiment, run one dataset, one fold, one learning rate, and a small number of epochs:

```powershell
python -m src.network.network_train_torch_manual `
  --datasets CAR `
  --losses ce ovr ovo `
  --criteria macro_f1 `
  --folds 1 `
  --learning-rates 0.001 `
  --max-epochs 3 `
  --device cpu `
  --output-dir src/network/results/smoke
```

This command is only a pipeline check. Its results must not be used as the final paper results because it runs only one fold, one learning rate, and three epochs.

## Full Experiment

To run all datasets, all three losses, both selection criteria, and all five folds:

```powershell
python -m src.network.network_train_torch_manual `
  --datasets ALL `
  --losses ce ovr ovo `
  --criteria accuracy macro_f1 `
  --folds 1 2 3 4 5 `
  --max-epochs 40 `
  --batch-size 8 `
  --patience 10 `
  --seed 42 `
  --device cuda `
  --output-dir src/network/results/neural_network_torch_manual_v2
```

If CUDA is unavailable, replace `--device cuda` with `--device cpu`. The full experiment may take substantially longer on CPU.

The full configuration contains:

```text
17 datasets × 3 losses × 2 selection criteria × 5 outer folds
```

## What the Training Script Does

For each dataset, loss, selection criterion, and outer fold, the script:

1. Loads the dataset through `src.datasets.load_dataset()`;
2. Obtains the outer folds through `src.split.get_outer_folds()`;
3. Selects an inner training and validation split;
4. Fits preprocessing on the inner training data only;
5. Searches the candidate learning rates;
6. Trains the MLP with explicit PyTorch batches and epochs;
7. Evaluates the selected configuration on the untouched outer test fold;
8. Refits the preprocessing and model on the complete outer training portion;
9. Predicts the outer test fold;
10. Saves the fold history, metrics, configuration, and predictions.

The test fold is not used for learning-rate selection or early stopping.

## Losses

`paper_losses.py` defines:

```text
ce  - Cross-Entropy loss
ovr - OVR-oriented loss
ovo - OVO-oriented loss
```

All three losses use the same `PaperMLP` output. The difference is the training objective, not the network architecture or prediction rule.

## Initial Results

The training script writes initial experiment results to the directory given by `--output-dir`.

For example:

```text
src/network/results/neural_network_torch_manual_v2/
├── CAR_ce_macro_f1_folds12345_seed42.json
├── CAR_ce_macro_f1_folds12345_seed42_predictions.csv
├── CAR_ovr_macro_f1_folds12345_seed42.json
├── CAR_ovr_macro_f1_folds12345_seed42_predictions.csv
└── ...
```

### JSON report

Each JSON report contains:

- Dataset and loss configuration;
- Selection criterion;
- Learning-rate candidates;
- Selected learning rate for every fold;
- Best epoch for every fold;
- Training history;
- Fold metrics;
- Combined test metrics;
- Class-label mapping;
- Split configuration and data fingerprint;
- Runtime and environment information where recorded.

### Prediction CSV

Each prediction CSV contains:

```text
sample_index
fold
y_true
```

These files are the primary raw prediction results. They can be used later to recompute metrics, confusion matrices, and per-class results without retraining.

## Analyzed and Summary Results

The training script itself produces raw JSON and prediction CSV files. Summary and analysis files are generated separately from those raw outputs.

Typical analyzed-result files are stored outside `src/network`, for example:

```text
src/network/results/
├── neural_network_torch_manual_v2/  # Initial raw experiment results
├── analysis/                        # Analyzed and summarized results
│   ├── summary_metrics.csv
│   ├── overall_means.csv
│   ├── imbalance_group_means.csv
│   ├── fold_details.csv
│   └── NN_results_summary.xlsx
├── raw_reports/                     # Optional copies of raw reports
└── audit_report.json
```

The distinction is:

| Result type | Typical files | Purpose |
|---|---|---|
| Initial/raw results | `*.json`, `*_predictions.csv` | Preserve each run's configuration, history, predictions, and fold information |
| Fold-level metrics | `fold_details.csv`, `metrics_by_fold.csv` | Inspect individual outer folds |
| Dataset-level summary | `summary_metrics.csv` | Compare each dataset, loss, and selection criterion |
| Overall analysis | `overall_means.csv` | Calculate the equal-weight mean across datasets |
| Imbalance-group analysis | `imbalance_group_means.csv` | Compare low, medium, and high IR groups |
| Presentation/report output | `NN_results_summary.xlsx`, figures | Prepare tables and figures for the report |
| Audit output | `audit_report.json` | Check completeness and internal consistency |

Do not overwrite raw prediction files with analyzed tables. Raw results should remain unchanged so that all later analysis can be reproduced.

## Result Analysis Rules

Use the same rules as the paper and project protocol:

- Compare the same dataset and the same selection criterion;
- Keep Accuracy-selected and Macro-F1-selected results separate;
- Merge the five outer-fold predictions before calculating the final dataset-level metrics;
- Do not average fold Macro-F1 or G-mean as a substitute for merged-prediction metrics;
- Calculate overall means as an equal-weight mean over datasets;
- Use `IR < 3`, `3 <= IR < 9`, and `IR >= 9` for imbalance groups;
- Keep class-label mappings consistent when creating confusion matrices.

## Tools

### Check Results

`tools/check_results.py` audits completed experiment outputs. It does not train models.

Run:

```powershell
python -m src.network.tools.check_results
```

The audit checks items such as:

- Expected experiment combinations;
- Complete five-fold coverage;
- Duplicate or missing predictions;
- Valid class IDs;
- Metrics recomputed from predictions;
- Fold records and learning-rate trials;
- Dataset fingerprints and split configuration.

### Reproduce or Export a Split

`tools/reproduce_split.py` reconstructs a split from a saved JSON report and can export sample membership.

Run:

```powershell
python -m src.network.tools.reproduce_split `
  --report src/network/results/neural_network_torch_manual_v2/CAR_ovo_macro_f1_folds12345_seed42.json `
  --fold 1
```

It does not train a model or modify the original result report.

## Archive

The `archive/` directory contains previous implementations, such as the former `skorch` version. These files are kept for historical comparison and should not be used for new final experiments.

The active training entry point is:

```text
network_train_torch_manual.py
```
