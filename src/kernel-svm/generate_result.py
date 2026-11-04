import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# =====================================================================
# Settings
# =====================================================================
RESULT_ROOT = os.path.dirname(os.path.abspath(__file__)) # folder that contains svm-predict-{monitor}/
MODEL_NAME = "svm"
MONITORS = ["Accuracy", "MacroF1"]
TRAIN_METHODS = ["one-versus-rest", "one-versus-one"] 

USE_OVO_TO_OVR_FOR_SVM = True   # Set to False to evaluate the OVO votes (the saved "predict") directly.

METRIC_NAMES = ["MacroF1", "Gmean", "Accuracy", "BalancedAccuracy", "OVOAUC"]

# =====================================================================
# 1. helpers
# =====================================================================
def softmax(x):
    """Softmax for each row of x (-inf entries become probability 0)."""
    row_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - row_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def split_target_by_fold(target, decision_val_folds):
    """Cut the pooled target back into one array per outer fold."""
    fold_sizes = [len(fold["scores"]) for fold in decision_val_folds]
    cut_points = np.cumsum(fold_sizes)[:-1]
    return np.split(np.asarray(target, dtype=int), cut_points)


# =====================================================================
# 2. Metrics that only need predictions and targets
# =====================================================================
def compute_basic_metrics(predict, target):
    """Accuracy, MacroF1, Gmean, BalancedAccuracy (all in percent, 2 decimals).
    Classes are the labels that appear in target."""
    predict = np.asarray(predict, dtype=int)
    target = np.asarray(target, dtype=int)
    classes = np.unique(target)

    f1_per_class = []
    recall_per_class = []
    for cls in classes:
        true_positive = np.sum((target == cls) & (predict == cls))
        false_positive = np.sum((target != cls) & (predict == cls))
        false_negative = np.sum((target == cls) & (predict != cls))

        # denominators are > 0 because every class in `classes` occurs in target
        f1_per_class.append(2 * true_positive /
                            (2 * true_positive + false_positive + false_negative))
        recall_per_class.append(true_positive / (true_positive + false_negative))

    recall_per_class = np.array(recall_per_class)

    # Gmean = geometric mean of per-class recall (0 if any recall is 0)
    with np.errstate(divide="ignore"):
        gmean = np.exp(np.log(recall_per_class).mean())

    return {
        "-Accuracy": round(float(accuracy_score(target, predict)) * 100, 2),
        "-MacroF1": round(float(np.mean(f1_per_class)) * 100, 2),
        "-Gmean": round(float(gmean) * 100, 2),
        "-BalancedAccuracy": round(float(balanced_accuracy_score(target, predict)) * 100, 2),
    }


# =====================================================================
# 3. OVR: AUC from the "labels" / "scores" of each fold
# =====================================================================
def build_ovr_score_matrix(decision_val_folds, classes):
    """Stack the per-fold OVR scores into one (num_samples x num_classes) matrix
    whose columns follow the sorted class order.
    Each fold may list its labels in a different order, so we map by label value.
    A class missing from a fold's training data gets -inf (never predicted)."""
    class_to_column = {int(cls): column for column, cls in enumerate(classes)}
    blocks = []

    for fold in decision_val_folds:
        scores = np.asarray(fold["scores"], dtype=float)
        block = np.full((scores.shape[0], len(classes)), -np.inf)
        for fold_column, label in enumerate(fold["labels"]):
            block[:, class_to_column[int(label)]] = scores[:, fold_column]
        blocks.append(block)

    return np.vstack(blocks)


def compute_ovr_auc(target, decision_val_folds):
    classes = np.unique(target)
    score_matrix = build_ovr_score_matrix(decision_val_folds, classes)
    probabilities = softmax(score_matrix)
    return round(roc_auc_score(target, probabilities, multi_class="ovo") * 100, 2)


# =====================================================================
# 4. OVO: pairwise AUC and OVO -> OVR conversion
# =====================================================================
def build_pairwise_data(decision_val_folds, target_per_fold):
    """Collect for every label pair (low, high), the pooled targets and scores.
    Scores are oriented so that a positive value means 'class low', no matter
    which order the pair had in a given fold."""
    pair_data = defaultdict(lambda: {"target": [], "score": []})

    for fold, fold_target in zip(decision_val_folds, target_per_fold):
        scores = np.asarray(fold["scores"], dtype=float)

        for column, (first_label, second_label) in enumerate(fold["pairs"]):
            low_label = min(first_label, second_label)
            high_label = max(first_label, second_label)

            if first_label == low_label:
                oriented_score = scores[:, column]      # positive -> first (= low)
            else:
                oriented_score = -scores[:, column]     # positive -> first (= high), so flip

            pair_data[(low_label, high_label)]["target"].append(fold_target)
            pair_data[(low_label, high_label)]["score"].append(oriented_score)

    return pair_data


def compute_ovo_auc(target_per_fold, decision_val_folds):
    """for each pairwise classifier, AUC on the samples of its two
    classes, then the mean over all pairs."""
    pair_data = build_pairwise_data(decision_val_folds, target_per_fold)
    auc_values = []

    for (low_label, high_label), data in sorted(pair_data.items()):
        pair_target = np.concatenate(data["target"])
        pair_score = np.concatenate(data["score"])

        keep = (pair_target == low_label) | (pair_target == high_label)
        is_low_class = (pair_target[keep] == low_label).astype(int)

        if is_low_class.sum() == 0 or is_low_class.sum() == len(is_low_class):
            continue    # only one class present, AUC undefined

        auc_values.append(roc_auc_score(is_low_class, pair_score[keep]))

    return round(float(np.mean(auc_values)) * 100, 2)

def convert_ovo_to_ovr_scores(decision_val_folds, classes):
    """Turn pairwise decision values into one score per class, per sample.
    ASSUMPTION: this is the same conversion as scikit-learn's
    _ovr_decision_function (votes + a small confidence term that breaks ties):
        score[class] = votes[class] + conf[class] / (3 * (|conf[class]| + 1))
    where conf[i] += d and conf[j] -= d for a pair (i, j) with decision value d.
    """
    class_to_column = {int(cls): column for column, cls in enumerate(classes)}
    num_classes = len(classes)
    blocks = []

    for fold in decision_val_folds:
        scores = np.asarray(fold["scores"], dtype=float)
        num_samples = scores.shape[0]
        votes = np.zeros((num_samples, num_classes))
        confidence = np.zeros((num_samples, num_classes))

        for column, (first_label, second_label) in enumerate(fold["pairs"]):
            first_col = class_to_column[int(first_label)]
            second_col = class_to_column[int(second_label)]
            decision = scores[:, column]

            confidence[:, first_col] += decision
            confidence[:, second_col] -= decision
            votes[decision > 0, first_col] += 1      # positive -> first label
            votes[decision <= 0, second_col] += 1    # otherwise second label

        # Only one class in the training data: no pairs, predict that class
        if len(fold["labels"]) == 1:
            votes[:, class_to_column[int(fold["labels"][0])]] = 1

        block = votes + confidence / (3 * (np.abs(confidence) + 1))

        # Classes missing from this fold's training data can never be predicted
        for cls, column in class_to_column.items():
            if cls not in [int(label) for label in fold["labels"]]:
                block[:, column] = -np.inf

        blocks.append(block)

    return np.vstack(blocks)


# =====================================================================
# 5. All metrics for one (monitor, method, dataset)
# =====================================================================
def get_metrics(predict, target, decision_val_folds, train_method, ovo2ovr=False):
    target = np.asarray(target, dtype=int)
    metrics = compute_basic_metrics(predict, target)

    ovo_auc = -999
    if decision_val_folds:
        if train_method == "one-versus-rest":
            ovo_auc = compute_ovr_auc(target, decision_val_folds)
        else:
            target_per_fold = split_target_by_fold(target, decision_val_folds)
            ovo_auc = compute_ovo_auc(target_per_fold, decision_val_folds)

            if ovo2ovr:
                # Re-predict from the converted OVR scores and recompute the metrics,
                # The OVO AUC above is kept
                classes = np.unique(target)
                ovr_scores = convert_ovo_to_ovr_scores(decision_val_folds, classes)
                new_predict = classes[np.argmax(ovr_scores, axis=1)]
                metrics = compute_basic_metrics(new_predict, target)

    metrics["-OVOAUC"] = ovo_auc
    return metrics


# =====================================================================
# 6. Loading the result files
# =====================================================================
def load_model_results(datasets, train_methods, monitors, model):
    """Read every 'test_*_k label.json' (outer fold k) and pool the folds.
    target/predict are concatenated; decision_val and config are kept per fold."""
    outputs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for dataset_name in tqdm(datasets):
        for monitor in monitors:
            for train_method in train_methods:
                result_dir = os.path.join(
                    RESULT_ROOT, f"{model}-predict-{monitor}", dataset_name, train_method)
                json_files = sorted(glob.glob(os.path.join(result_dir, "*.json")))
                if len(json_files) == 0:
                    raise FileNotFoundError(f"No result files in {result_dir}")

                pooled = {"target": [], "predict": [], "decision_val": [], "config": []}
                for json_file in json_files:
                    with open(json_file, "r") as file:
                        info = json.load(file)
                    pooled["target"].extend(int(v) for v in info["target"])
                    pooled["predict"].extend(int(v) for v in info["predict"])
                    pooled["decision_val"].append(info["decision_val"])   # one dict per fold
                    pooled["config"].append(info["config"])

                outputs[monitor][train_method][dataset_name] = pooled

            # sanity check: OVR and OVO must have been evaluated on the same targets
            if "one-versus-rest" in train_methods and "one-versus-one" in train_methods:
                ovr_target = outputs[monitor]["one-versus-rest"][dataset_name]["target"]
                ovo_target = outputs[monitor]["one-versus-one"][dataset_name]["target"]
                assert ovr_target == ovo_target, f"target mismatch for {dataset_name}"

    return outputs


# =====================================================================
# 7. Metric tables
# =====================================================================
def generate_metrics(dataset_outputs, model, use_prefix3=True):
    dataset_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    metric_df = defaultdict(lambda: defaultdict(dict))

    for monitor in dataset_outputs.keys():
        for train_method in dataset_outputs[monitor].keys():
            for dataset_name in tqdm(dataset_outputs[monitor][train_method].keys()):
                # SVM OVO gives K*(K-1)/2 values, so it is converted to OVR scores
                ovo2ovr = (model == "svm"
                           and train_method == "one-versus-one"
                           and USE_OVO_TO_OVR_FOR_SVM)

                data = dataset_outputs[monitor][train_method][dataset_name]
                metrics = get_metrics(predict=data["predict"],
                                      target=data["target"],
                                      decision_val_folds=data["decision_val"],
                                      train_method=train_method,
                                      ovo2ovr=ovo2ovr)

                metric_info = [float(metrics["-MacroF1"]), float(metrics["-Gmean"]),
                               float(metrics["-Accuracy"]), float(metrics["-BalancedAccuracy"]),
                               float(metrics["-OVOAUC"])]

                table_name = dataset_name[:3] if use_prefix3 else dataset_name
                dataset_metrics[monitor][train_method][table_name] = metric_info

            metric_df[monitor][train_method] = pd.DataFrame(
                dataset_metrics[monitor][train_method], index=METRIC_NAMES).T

        metric_df[monitor] = pd.concat(metric_df[monitor], axis=1)
        metric_df[monitor].columns = metric_df[monitor].columns.swaplevel(0, 1)
        metric_df[monitor] = metric_df[monitor].reindex(
            columns=["Accuracy", "MacroF1", "Gmean", "BalancedAccuracy", "OVOAUC"], level=0)
        metric_df[monitor].loc["Average"] = metric_df[monitor].mean()
        metric_df[monitor] = metric_df[monitor].round(2)

    return metric_df, dataset_metrics


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":

    # datasets = ["AUT", "BAL"]
    
    datasets =  [   'AUT', 'BAL', 'CAR', 'CLE','DER', 
                    'ECO', 'FLA', 'GLA', 'HAY', 
                    'HCV', 'LYM', 'NEW', 'SEG', 
                    'SHU', 'THY', 'VEH', 'ZOO'    ]

    dataset_outputs = load_model_results(datasets, TRAIN_METHODS, MONITORS, MODEL_NAME)
    metric_df, dataset_metrics = generate_metrics(dataset_outputs, MODEL_NAME)

    # destination_dir = os.path.join("results", MODEL_NAME)
    destination_dir = os.path.join(RESULT_ROOT, "results", MODEL_NAME)
    os.makedirs(destination_dir, exist_ok=True)

    with open(os.path.join(destination_dir, "dataset_outputs.json"), "w") as file:
        json.dump(dataset_outputs, file, indent=4)
    with open(os.path.join(destination_dir, "dataset_metrics.json"), "w") as file:
        json.dump(dataset_metrics, file, indent=4)
    for monitor in metric_df.keys():
        metric_df[monitor].to_csv(os.path.join(destination_dir, f"{monitor}.csv"))
        print(f"\n=== {monitor} ===")
        print(metric_df[monitor])