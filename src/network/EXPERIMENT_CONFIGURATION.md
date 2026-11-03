# Neural Network Experiment Configuration

This file documents the fixed settings used by `network_train_torch_manual.py`. They are intentionally kept out of the per-run JSON metadata because they describe the common experiment protocol rather than dataset-specific results.

## Implementation

```text
PyTorch manual training pipeline
```

The training loop explicitly performs:

1. Forward propagation;
2. Loss calculation;
3. Backpropagation;
4. Adam parameter updates;
5. Validation;
6. Early stopping;
7. Best-epoch selection;
8. Outer-test prediction.

## Model

The paper MLP is:

```text
Linear(input_dim → 128)
→ LayerNorm(128)
→ ReLU
→ Dropout(0.1)
→ Linear(128 → 64)
→ LayerNorm(64)
→ ReLU
→ Dropout(0.1)
→ Linear(64 → number_of_classes)
```

The same network architecture and `argmax` prediction rule are used for CE, OVR, and OVO.

## Losses

The training configuration supports:

```text
ce  = Cross-Entropy loss
ovr = OVR-oriented loss
ovo = OVO-oriented loss
```

The loss implementation is in:

```text
src/network/paper_losses.py
```

## Fixed Experiment Constants

| Setting | Value |
|---|---|
| Losses | `ce`, `ovr`, `ovo` |
| Selection criteria | `accuracy`, `macro_f1` |
| Outer folds | 1, 2, 3, 4, 5 |
| Learning rates | `1e-5`, `5e-5`, `1e-4`, `5e-4`, `1e-3`, `5e-3`, `1e-2`, `5e-2` |
| Optimizer | `torch.optim.Adam` |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-8` |
| Weight decay | `0` |
| Batch size | `8` |
| Maximum epochs | `40` |
| Early-stopping patience | `10` epochs |
| Random seed | `42` |
| Default thread count | `1` |
| Main metrics | Accuracy, Macro-F1, G-mean, Balanced Accuracy |

## Data Processing

- Numerical features are scaled to `[0,1]`;
- Categorical features use one-hot encoding;
- Missing numerical values use the training-partition median;
- Missing categorical values use the training-partition most frequent value;
- Preprocessing is fitted on training data only;
- Validation and test data are transformed using the fitted preprocessing pipeline.

## Evaluation Protocol

- Use stratified outer five-fold evaluation;
- Hold one outer fold out for testing;
- Use the remaining four folds for training and hyperparameter selection;
- Use the first inner split for neural-network learning-rate selection, following the paper's reduced-cost setting;
- Use the selected learning rate and selected best epoch to refit on the complete outer training portion;
- Never use the outer test fold for learning-rate selection or early stopping;
- Merge all outer-test predictions before calculating the final dataset-level metrics;
- Explicitly include all dataset classes in metric calculations;
- Zero denominators are handled with zero values.

## Refit Policy

The paper does not fully specify early stopping during final refitting. This implementation uses the best epoch selected on the inner validation split and retrains on the complete outer training portion for that number of epochs.

## OVO Class-Count Scaling

Training still uses mini-batches of size 8. The OVO loss uses class counts
from the complete training partition:

```text
sum(pair_loss / (|O_y| + |O_j|)) / C(k,2)
```

Here, `|O_y|` and `|O_j|` are fixed counts from the complete training partition. The mini-batch supplies the current examples and gradients, while the denominator does not change with the batch.

This reduction must be validated using the paper equations and small hand-calculated test cases before the final full run.

## Result Metadata

Dataset-specific metadata remains in each JSON result file:

- Dataset;
- Loss;
- Selection criterion;
- Class labels;
- Seed;
- Split configuration;
- Learning-rate candidates;
- Fold selection;
- Completion status;
- Evaluated sample count;
- Combined test metrics;
- Fold-level results.

Common fixed settings are documented in this file rather than repeated in every result JSON.
