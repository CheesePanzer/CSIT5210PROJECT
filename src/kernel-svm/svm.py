import os
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from datasets import load_dataset
from split import get_outer_folds, get_inner_subsets
from preprocessing import preprocess_partition

# =====================================================================
# Settings
# =====================================================================
LOG2_C_VALUES = list(range(-2, 13))        # C     = 2^-2  ... 2^12  (15 values)
LOG2_GAMMA_VALUES = list(range(-10, 5))    # gamma = 2^-10 ... 2^4   (15 values)

MONITORS = ["Accuracy", "MacroF1"]
SCHEMES = ["ovr", "ovo"]

REPLICATE_OVO_ROUNDING_QUIRK = True        # set to False to use the exact grid values instead

NUM_PARALLEL_JOBS = -1                     # -1 = use all CPU cores
OUTPUT_ROOT = "./kernel-svm"

# =====================================================================
# 1. Model and metrics
# =====================================================================
def create_svm(C, gamma):
    """SVC configured like LIBSVM's default C-SVC with an RBF kernel (-t 2)."""
    return SVC(
        C=C,
        kernel="rbf",
        gamma=gamma,
        tol=1e-3,              # LIBSVM -e default
        shrinking=True,        # LIBSVM -h default
        cache_size=100,        # LIBSVM -m default
        probability=False,     # LIBSVM -b default
        class_weight=None,     # LIBSVM -w default
    )
    
def compute_macro_f1(y_true, y_pred):
    """Macro-F1 over every class that appears in y_true or y_pred.
    A class with tp = fp = fn = 0 is skipped"""
    all_classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_per_class = []

    for cls in all_classes:
        true_positive = np.sum((y_true == cls) & (y_pred == cls))
        false_positive = np.sum((y_true != cls) & (y_pred == cls))
        false_negative = np.sum((y_true == cls) & (y_pred != cls))

        if true_positive + false_positive + false_negative == 0:
            continue

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0.0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_per_class.append(f1)

    return float(np.mean(f1_per_class))


def compute_score(y_true, y_pred, monitor, scheme):
    """Score in percent:
       - OVR: round to 2 decimals
       - OVO: printed with C's %g, i.e. 6 significant digits."""
    if monitor == "Accuracy":
        value = 100.0 * np.mean(y_true == y_pred)
    else:
        value = 100.0 * compute_macro_f1(y_true, y_pred)

    if scheme == "ovr":
        return round(value, 2)
    return float("%g" % value)

# =====================================================================
# 2. One-vs-Rest
# =====================================================================
def ovr_train_and_predict(X_train, y_train, X_test, C, gamma):
    """Train one binary SVM per label using sklearn's built-in OneVsRestClassifier.
    
    Returns: predictions, decision values (n_test x n_labels), label order."""
    
    # 1. Handle degenerate case: only one class in the training data
    unique_labels = np.unique(y_train)
    if len(unique_labels) == 1:
        predictions = np.full(len(X_test), unique_labels[0])
        decision_values = np.zeros((len(X_test), 1))
        return predictions, decision_values, unique_labels

    # 2. Wrap the base SVC inside OneVsRestClassifier
    base_svm = create_svm(C, gamma)
    ovr_clf = OneVsRestClassifier(base_svm)
    
    # 3. Train the model
    ovr_clf.fit(X_train, y_train)
    
    # 4. Predict labels and get raw decision values
    predictions = ovr_clf.predict(X_test)
    decision_values = ovr_clf.decision_function(X_test)
    
    # 5. Extract the label order used internally by sklearn (sorted by default)
    labels = ovr_clf.classes_

    return predictions, decision_values, labels

# def get_labels_in_first_appearance_order(y):
#     """Labels in the order they first appear in y."""
#     return list(dict.fromkeys(y.tolist()))

# def ovr_train_and_predict(X_train, y_train, X_test, C, gamma):
#     """Train one binary SVM per label (label = +1, all others = -1).
#     Predict the label whose SVM gives the largest decision value.
#     Returns: predictions, decision values (n_test x n_labels), label order."""

#     labels = get_labels_in_first_appearance_order(y_train)

#     # Degenerate case: only one class in the training data
#     if len(labels) == 1:
#         predictions = np.full(len(X_test), labels[0])
#         decision_values = np.zeros((len(X_test), 1))
#         return predictions, decision_values, labels

#     decision_columns = []
#     for label in labels:
#         binary_targets = np.where(y_train == label, 1, -1)
#         svm = create_svm(C, gamma).fit(X_train, binary_targets)
#         # decision_function > 0 means "+1", i.e. "this label"
#         decision_columns.append(svm.decision_function(X_test))

#     decision_values = np.column_stack(decision_columns)

#     # argmax returns the FIRST maximum, so ties go to the earliest label.
#     # This matches the reference's stable sort.
#     best_column = np.argmax(decision_values, axis=1)
#     predictions = np.array(labels)[best_column]

#     return predictions, decision_values, labels

# =====================================================================
# 3. One-vs-One
# =====================================================================
def ovo_train_and_predict(
    X_train, y_train, X_test, C, gamma
):
    svm = SVC(
        C=C,
        kernel="rbf",
        gamma=gamma,
        tol=1e-3,
        shrinking=True,
        cache_size=100,
        probability=False,
        class_weight=None,
        decision_function_shape="ovo"
    )

    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)
    decision_values = svm.decision_function(X_test)

    return predictions, decision_values, svm.classes_

def train_and_predict(scheme, X_train, y_train, X_test, C, gamma):
    """Dispatch to the OVR or OVO implementation."""
    if scheme == "ovr":
        return ovr_train_and_predict(X_train, y_train, X_test, C, gamma)
    return ovo_train_and_predict(X_train, y_train, X_test, C, gamma)

# def get_libsvm_label_order(y):
#     """Label order used internally by LIBSVM (svm_group_classes):
#     first-appearance order, except that for a 2-class problem whose labels
#     appear as (-1, +1), LIBSVM swaps them to (+1, -1)."""
#     labels = get_labels_in_first_appearance_order(y)
#     if len(labels) == 2 and labels[0] == -1 and labels[1] == 1:
#         labels.reverse()
#     return labels

# def ovo_train_and_predict(X_train, y_train, X_test, C, gamma):
#     """Train K*(K-1)/2 pairwise SVMs and predict by voting.
#     LIBSVM:
#     For pair (i, j) with i < j, class i is +1 and class j is -1.
#     A positive decision value votes for i, otherwise for j.
#     On equal votes the first label wins.
#     Returns: predictions, pairwise decision values (n_test x K(K-1)/2), label order."""
#     labels = get_libsvm_label_order(y_train)
#     num_labels = len(labels)

#     # Degenerate case: only one class in the training data
#     if num_labels == 1:
#         predictions = np.full(len(X_test), labels[0])
#         decision_values = np.zeros((len(X_test), 0))
#         return predictions, decision_values, labels

#     rows_of_label = [np.flatnonzero(y_train == label) for label in labels]
#     votes = np.zeros((len(X_test), num_labels), dtype=int)
#     decision_columns = []

#     for i in range(num_labels):
#         for j in range(i + 1, num_labels):
#             # Class i rows first, then class j rows (the order LIBSVM uses)
#             pair_rows = np.concatenate([rows_of_label[i], rows_of_label[j]])
#             pair_targets = np.concatenate([
#                 np.ones(len(rows_of_label[i])),
#                 -np.ones(len(rows_of_label[j])),
#             ])

#             svm = create_svm(C, gamma).fit(X_train[pair_rows], pair_targets)
#             decision = svm.decision_function(X_test)
            
#             decision_columns.append(decision)

#             votes[decision > 0, i] += 1
#             votes[decision <= 0, j] += 1

#     best_column = np.argmax(votes, axis=1)    # first maximum wins
#     predictions = np.array(labels)[best_column]
#     decision_values = np.column_stack(decision_columns)

#     return predictions, decision_values, labels

# =====================================================================
# 4. Order in which LIBSVM grid visits (C, gamma) pairs (used for OVO tie-breaking)
# =====================================================================
def permute_sequence(sequence):
    """Middle element first, then the left half, then the right half (recursively)."""
    n = len(sequence)
    if n <= 1:
        return sequence
    middle = n // 2
    left = permute_sequence(sequence[:middle])
    right = permute_sequence(sequence[middle + 1:])
    return [sequence[middle]] + left + right

def get_grid_py_job_order():
    """List of (log2C, log2gamma) pairs in the order grid evaluates
    (it grows the grid outward from the centre)."""
    c_sequence = permute_sequence(LOG2_C_VALUES)
    gamma_sequence = permute_sequence(LOG2_GAMMA_VALUES)
    num_c = float(len(c_sequence))
    num_gamma = float(len(gamma_sequence))

    c_index = 0
    gamma_index = 0
    jobs = []

    while c_index < num_c or gamma_index < num_gamma:
        if c_index / num_c < gamma_index / num_gamma:
            # Add one new C value, paired with every gamma used so far
            for k in range(gamma_index):
                jobs.append((c_sequence[c_index], gamma_sequence[k]))
            c_index += 1
        else:
            # Add one new gamma value, paired with every C used so far
            for k in range(c_index):
                jobs.append((c_sequence[k], gamma_sequence[gamma_index]))
            gamma_index += 1

    return jobs

# =====================================================================
# 5. Inner cross-validation on the 5 subsets from get_inner_subsets
# =====================================================================
def prepare_inner_folds(X, y, train_val_idx, categorical_cols):
    """Preprocess each of the 5 inner (train, validation) subsets ONCE,
    so the same arrays are reused for all 225 (C, gamma) pairs.

    Returns a list of (X_inner_train, y_inner_train, X_inner_val, y_inner_val)."""
    inner_folds = []

    for inner_train_rel, inner_val_rel in get_inner_subsets(X, train_val_idx, y):
        inner_train_idx = train_val_idx[inner_train_rel]
        inner_val_idx = train_val_idx[inner_val_rel]

        X_inner_train, X_inner_val = preprocess_partition(
            X, inner_train_idx, inner_val_idx, categorical_cols)
        y_inner_train = y[inner_train_idx]
        y_inner_val = y[inner_val_idx]

        inner_folds.append((X_inner_train, y_inner_train, X_inner_val, y_inner_val))

    return inner_folds


def cross_validate_one_pair(inner_folds, scheme, log2_c, log2_gamma):
    """Run the inner CV for one (C, gamma) pair.
    Validation predictions from all 5 folds are pooled, then scored once. 
    One fit gives both the Accuracy and the MacroF1 score, 
    which is the same as running the two searches separately."""
    C = 2.0 ** log2_c
    gamma = 2.0 ** log2_gamma

    all_true = []
    all_pred = []
    for X_inner_train, y_inner_train, X_inner_val, y_inner_val in inner_folds:
        predictions, _, _ = train_and_predict(
            scheme, X_inner_train, y_inner_train, X_inner_val, C, gamma)
        all_true.append(y_inner_val)
        all_pred.append(predictions)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    return {monitor: compute_score(all_true, all_pred, monitor, scheme)
            for monitor in MONITORS}


def evaluate_whole_grid(inner_folds, scheme):
    """Inner-CV scores for all 225 pairs (in parallel).
    Returns {(log2_c, log2_gamma): {"Accuracy": ..., "MacroF1": ...}}."""
    pairs = [(log2_c, log2_gamma)
             for log2_c in LOG2_C_VALUES
             for log2_gamma in LOG2_GAMMA_VALUES]

    scores = Parallel(n_jobs=NUM_PARALLEL_JOBS)(
        delayed(cross_validate_one_pair)(inner_folds, scheme, log2_c, log2_gamma)
        for log2_c, log2_gamma in pairs)

    return dict(zip(pairs, scores))


# =====================================================================
# 6. Choosing the best (C, gamma)
# =====================================================================
def select_best_pair_ovr(grid_scores, monitor):
    """binary. search: C outer loop (ascending), gamma inner loop (ascending),
    update only on a strictly better score, starting from 0."""
    best_score = 0.0
    best_pair = (LOG2_C_VALUES[0], LOG2_GAMMA_VALUES[0])

    for log2_c in LOG2_C_VALUES:
        for log2_gamma in LOG2_GAMMA_VALUES:
            current_score = grid_scores[(log2_c, log2_gamma)][monitor]
            if current_score > best_score:
                best_score = current_score
                best_pair = (log2_c, log2_gamma)

    return best_pair, best_score


def select_best_pair_ovo(grid_scores, monitor):
    """grid search: visit pairs in LIBSVM grid's order. Update when the score is
    strictly better, or when it ties with the same gamma and a smaller C."""
    best_score = -1.0
    best_log2_c = None
    best_log2_gamma = None

    for log2_c, log2_gamma in get_grid_py_job_order():
        current_score = grid_scores[(log2_c, log2_gamma)][monitor]

        strictly_better = current_score > best_score
        tie_with_smaller_c = (current_score == best_score
                              and log2_gamma == best_log2_gamma
                              and log2_c < best_log2_c)

        if strictly_better or tie_with_smaller_c:
            best_score = current_score
            best_log2_c = log2_c
            best_log2_gamma = log2_gamma

    return (best_log2_c, best_log2_gamma), best_score


def get_final_hyperparameters(scheme, log2_c, log2_gamma):
    """Convert the chosen exponents to (C, gamma) for the final model.
    For OVO this optionally reproduces '%.1f' / '%.6g' rounding."""
    C = 2.0 ** log2_c
    gamma = 2.0 ** log2_gamma

    if scheme == "ovo" and REPLICATE_OVO_ROUNDING_QUIRK:
        C = float("%.1f" % C)            # 0.25 -> 0.2
        gamma = float("%.6g" % gamma)    # 2^-10 -> 0.000976562

    return C, gamma


# =====================================================================
# 7. Writing the output files
# =====================================================================
def write_output_files(monitor, dataset_code, scheme, fold_index,
                       y_test, predictions, decision_values, labels, config_string):
    """Write the .svm.out (predictions) and label.json files"""
    folder_name = "one-versus-rest" if scheme == "ovr" else "one-versus-one"  
    folder = os.path.join(OUTPUT_ROOT, f"svm-predict-{monitor}", dataset_code, folder_name)
    os.makedirs(folder, exist_ok=True)

    # Predictions file: one predicted label per line
    out_path = os.path.join(folder, f"train_{dataset_code}_{fold_index}.svm.out")
    with open(out_path, "w") as out_file:
        out_file.write("\n".join(str(p) for p in predictions.tolist()) + "\n")

    # Decision values formatted per scheme.
    decision_data = []
    if scheme == "ovr":        
        decision_data = {
            "labels": [int(label) for label in labels],
            "scores": decision_values.tolist()
        }
    else:
        num_classes = len(labels)
        pairs = [
            [int(labels[i]), int(labels[j])]
            for i in range(num_classes)
            for j in range(i + 1, num_classes)
        ]
        decision_data = {
            "labels": [int(label) for label in labels],
            "pairs": pairs,
            "scores": decision_values.tolist()
        }
        
    json_path = os.path.join(folder, f"test_{dataset_code}_{fold_index} label.json")
    with open(json_path, "w") as json_file:
        json.dump({
            "target": y_test.tolist(),
            "predict": predictions.tolist(),
            "decision_val": decision_data,
            "config": config_string,
        }, json_file)

# =====================================================================
# 8. Driver
# =====================================================================
def run_dataset(dataset_code):
    X, y, meta = load_dataset(dataset_code, "../data/raw")
    y = np.asarray(y)
    categorical_cols = meta["categorical_cols"]

    outer_folds = get_outer_folds(X, y, metadata=meta)

    for fold_index, (train_val_idx, test_idx) in enumerate(outer_folds):
        train_val_idx = np.asarray(train_val_idx)
        test_idx = np.asarray(test_idx)

        # Inner subsets (preprocessed once) for hyperparameter search
        inner_folds = prepare_inner_folds(X, y, train_val_idx, categorical_cols)

        # Full training set / outer test set for the final model
        X_train, X_test = preprocess_partition(X, train_val_idx, test_idx, categorical_cols)
        y_train, y_test = y[train_val_idx], y[test_idx]

        for scheme in SCHEMES:
            grid_scores = evaluate_whole_grid(inner_folds, scheme)

            for monitor in MONITORS:
                # 1) pick the best (C, gamma) for this monitor
                if scheme == "ovr":
                    (log2_c, log2_gamma), cv_score = select_best_pair_ovr(grid_scores, monitor)
                else:
                    (log2_c, log2_gamma), cv_score = select_best_pair_ovo(grid_scores, monitor)

                # 2) train the final model on the whole training part, predict the test fold
                C, gamma = get_final_hyperparameters(scheme, log2_c, log2_gamma)
                predictions, decision_values, labels = train_and_predict(
                    scheme, X_train, y_train, X_test, C, gamma)

                # 3) save results
                if scheme == "ovo":
                    config_string = f"-c {C:.1f} -t 2 -g {gamma:.6g}"
                else:
                    config_string = f"-c {C} -t 2 -g {gamma}"

                write_output_files(monitor, dataset_code, scheme, fold_index,
                                   y_test, predictions, decision_values, labels,
                                   config_string)

                test_score = compute_score(y_test, predictions, monitor, scheme)
                print(f"{dataset_code} fold{fold_index} {scheme} {monitor}: "
                      f"inner-CV={cv_score}  log2C={log2_c}  log2gamma={log2_gamma}  "
                      f"test={test_score}")


if __name__ == "__main__":
    
    # DATASET_CODES = ["AUT", "BAL"]   # for testing
    
    DATASET_CODES = [   'AUT', 'BAL', 'CAR', 'CLE',
                        'DER', 'ECO', 'FLA', 'GLA',
                        'HAY', 'HCV', 'LYM', 'NEW',
                        'SEG', 'SHU', 'THY', 'VEH', 'ZOO'
                    ]
    
    for dataset_code in DATASET_CODES:
        run_dataset(dataset_code)