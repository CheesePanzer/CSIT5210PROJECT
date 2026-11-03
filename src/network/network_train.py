"""论文神经网络实验。

从项目根目录运行：
    py -3.11 -m src.network.network_train --datasets CAR --losses ovo

PyTorch 仅负责网络组件、张量运算与自动求导。损失、Adam、批次循环、
早停、学习率搜索和指标均在本文件实现，不调用现成分类器或训练器。
数据读取/划分/预处理复用组员的 src 接口（这些接口依赖 sklearn）。
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.dataset_registry import get_available_datasets
from src.datasets import load_dataset
from src.preprocessing import preprocess_partition
from src.split import get_inner_subsets, get_outer_folds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PaperMLP(nn.Module):
    """三层 Linear；两个隐藏层均按 LayerNorm -> ReLU -> Dropout 排列。"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # 输出 logits；不要在网络末尾额外添加 sigmoid 或 softmax。
        return self.layers(x)


def classification_loss(logits, targets, loss_name, class_counts):
    """手写论文的式 (6)、(9)、(12)，返回一个批次的标量损失。

    class_counts 是本次训练分区的全量类别计数，不是当前 batch 的计数。
    OVO 的逐样本损失乘 N 后取 batch 均值，使随机 batch 梯度无偏估计
    论文式 (12) 的完整训练目标；保留有序类别对产生的系数 2。
    """
    target_logits = logits.gather(1, targets[:, None]).squeeze(1)
    log_normalizer = torch.logsumexp(logits, dim=1)

    if loss_name == "ce":
        return (log_normalizer - target_logits).mean()

    k = logits.shape[1]
    if k < 2:
        raise ValueError("Classification requires at least two classes.")
    class_ids = torch.arange(k, device=logits.device)
    is_target = class_ids[None, :] == targets[:, None]

    if loss_name == "ovr":
        # log(1-p_i) = logsumexp(f_j, j!=i) - logsumexp(f_j)。
        # 使用 log 空间计算，避免先 softmax 后 log(0) 导致数值溢出。
        exclude_self = torch.eye(k, dtype=torch.bool, device=logits.device)
        other_logits = logits[:, None, :].expand(-1, k, -1)
        other_logsum = torch.logsumexp(
            other_logits.masked_fill(exclude_self[None, :, :], -torch.inf), dim=2
        )
        negative_terms = log_normalizer[:, None] - other_logsum
        negative_terms = negative_terms.masked_fill(is_target, 0.0)
        return (log_normalizer - target_logits + negative_terms.sum(dim=1)).mean()

    if loss_name == "ovo":
        # -log p_(y,j) = log(1 + exp(f_j-f_y))。
        pair_losses = torch.logaddexp(
            torch.zeros_like(logits), logits - target_logits[:, None]
        )
        counts = class_counts.to(device=logits.device, dtype=logits.dtype)
        pair_sizes = counts[targets, None] + counts[None, :]
        if torch.any(pair_sizes <= 0):
            raise ValueError("OVO class counts must describe the training partition.")
        weighted = (pair_losses / pair_sizes).masked_fill(is_target, 0.0)
        return 2.0 * counts.sum() * weighted.sum(dim=1).mean()

    raise ValueError(f"Unknown loss: {loss_name}")


class ManualAdam:
    """手写 Adam，无权重衰减；beta/epsilon 是论文未注明的实现选择。"""

    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_number = 0
        self.first_moments = [torch.zeros_like(p) for p in self.parameters]
        self.second_moments = [torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        self.step_number += 1
        for parameter, first, second in zip(
            self.parameters, self.first_moments, self.second_moments
        ):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(
                gradient, gradient, value=1.0 - self.beta2
            )
            first_corrected = first / (1.0 - self.beta1 ** self.step_number)
            second_corrected = second / (1.0 - self.beta2 ** self.step_number)
            parameter.addcdiv_(
                first_corrected, second_corrected.sqrt() + self.eps, value=-self.lr
            )


def compute_metrics(y_true, y_pred, num_classes):
    """从混淆矩阵手写四个指标；未出现/未预测类别的零分母记为 0。

    始终保留数据集全部 k 类。少数类缺失时不静默删除该类。
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("Expected nonempty true/predicted labels of equal shape.")
    if np.any(y_true < 0) or np.any(y_true >= num_classes):
        raise ValueError("True labels are outside the class range.")
    if np.any(y_pred < 0) or np.any(y_pred >= num_classes):
        raise ValueError("Predicted labels are outside the class range.")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    tp = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(tp), where=(precision + recall) > 0,
    )
    g_mean = 0.0 if np.any(recall == 0) else float(np.exp(np.log(recall).mean()))
    return {
        "accuracy": float(tp.sum() / len(y_true)),
        "macro_f1": float(f1.mean()),
        "g_mean": g_mean,
        "ba": float(recall.mean()),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def train_one_epoch(model, features, labels, optimizer, loss_name, counts, batch_size):
    model.train()
    order = torch.randperm(len(labels), device=labels.device)
    total_loss = 0.0
    # 自己切 batch；包括最后一个不足 batch_size 的批次。
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        optimizer.zero_grad()
        loss = classification_loss(model(features[index]), labels[index], loss_name, counts)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss; check data and learning rate.")
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * len(index)
    return total_loss / len(labels)


@torch.no_grad()
def predict(model, features, batch_size=1024):
    model.eval()  # 验证/测试关闭 Dropout。
    predictions = []
    for start in range(0, len(features), batch_size):
        predictions.append(model(features[start:start + batch_size]).argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()


def as_tensors(features, labels, device):
    return (
        torch.as_tensor(features, dtype=torch.float32, device=device),
        torch.as_tensor(labels, dtype=torch.long, device=device),
    )


def select_learning_rate(X_train, y_train, X_val, y_val, num_classes,
                         loss_name, criterion, args, seed):
    train_x, train_y = as_tensors(X_train, y_train, args.device)
    val_x, _ = as_tensors(X_val, y_val, args.device)
    counts = torch.bincount(train_y, minlength=num_classes)
    trials = []
    for lr in args.learning_rates:
        # 每个学习率采用相同初始化和随机序列，便于比较。
        set_seed(seed)
        model = PaperMLP(X_train.shape[1], num_classes).to(args.device)
        optimizer = ManualAdam(model.parameters(), lr=lr)
        best_score, best_epoch, stale_epochs = -math.inf, 0, 0
        history = []
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(
                model, train_x, train_y, optimizer, loss_name, counts, args.batch_size
            )
            metrics = compute_metrics(y_val, predict(model, val_x), num_classes)
            score = metrics[criterion]
            history.append({"epoch": epoch, "train_loss": train_loss, "val_score": score})
            if score > best_score:
                best_score, best_epoch, stale_epochs = score, epoch, 0
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                break
        trials.append({
            "learning_rate": lr, "best_val_score": best_score,
            "best_epoch": best_epoch, "history": history,
        })
        print(f"    lr={lr:g} val_{criterion}={best_score:.4f} "
              f"best_epoch={best_epoch} ran={len(history)}", flush=True)
    # 分数相同保留候选列表中较早的学习率；epoch 同分保留较早轮次。
    best = max(trials, key=lambda trial: trial["best_val_score"])
    return best, trials


def run_experiment(dataset_id, loss_name, criterion, args):
    X, y, meta = load_dataset(dataset_id, data_dir=str(args.data_dir))
    folds = get_outer_folds(X, y, metadata=meta, seed=args.seed)
    k = meta["n_classes"]
    all_predictions = np.full(len(y), -1, dtype=np.int64)
    fold_assignment = np.full(len(y), -1, dtype=np.int64)
    fold_results = []

    for fold_number in args.folds:
        train_val_idx, test_idx = folds[fold_number - 1]
        inner_train, inner_val = get_inner_subsets(
            X, train_val_idx, y, seed=args.seed
        )[0]
        train_idx = train_val_idx[inner_train]
        val_idx = train_val_idx[inner_val]
        X_train, X_val = preprocess_partition(
            X, train_idx, val_idx, meta["categorical_cols"]
        )
        fold_seed = args.seed + fold_number
        print(f"[{dataset_id}/{loss_name}/{criterion}] fold {fold_number}/5 "
              f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
              flush=True)
        best, trials = select_learning_rate(
            X_train, y[train_idx], X_val, y[val_idx], k,
            loss_name, criterion, args, fold_seed,
        )

        # 调参后在外层全部训练数据重新拟合预处理器，输入维度可能变化。
        X_refit, X_test = preprocess_partition(
            X, train_val_idx, test_idx, meta["categorical_cols"]
        )
        set_seed(fold_seed)
        model = PaperMLP(X_refit.shape[1], k).to(args.device)
        refit_x, refit_y = as_tensors(X_refit, y[train_val_idx], args.device)
        test_x, _ = as_tensors(X_test, y[test_idx], args.device)
        counts = torch.bincount(refit_y, minlength=k)
        optimizer = ManualAdam(model.parameters(), lr=best["learning_rate"])
        # 论文未明确重训的早停方案。本实现采用内层选中的最佳 epoch 数，
        # 在整个外层训练分区重训；测试集不参与选学习率或训练轮数。
        for _ in range(best["best_epoch"]):
            train_one_epoch(
                model, refit_x, refit_y, optimizer, loss_name, counts, args.batch_size
            )
        predictions = predict(model, test_x)
        all_predictions[test_idx] = predictions
        fold_assignment[test_idx] = fold_number
        metrics = compute_metrics(y[test_idx], predictions, k)
        fold_results.append({
            "fold": fold_number, "seed": fold_seed,
            "inner_input_dim": X_train.shape[1], "refit_input_dim": X_refit.shape[1],
            "train_val_indices": train_val_idx.tolist(),
            "inner_train_indices": train_idx.tolist(), "inner_val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist(), "refit_class_counts": counts.cpu().tolist(),
            "selected_learning_rate": best["learning_rate"],
            "refit_epochs": best["best_epoch"], "trials": trials, "test_metrics": metrics,
        })
        print(f"    test accuracy={metrics['accuracy']:.4f} "
              f"macro_f1={metrics['macro_f1']:.4f}", flush=True)

    evaluated = all_predictions >= 0
    # 按论文合并各折预测后计算指标，不对五个 fold 的指标直接求平均。
    combined_metrics = compute_metrics(y[evaluated], all_predictions[evaluated], k)
    result = {
        "dataset": dataset_id, "loss": loss_name, "selection_criterion": criterion,
        "seed": args.seed, "class_labels": meta["class_labels"],
        "learning_rates": args.learning_rates, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "device": str(args.device), "torch_version": torch.__version__,
        "adam": {"betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0},
        "refit_policy": "inner_best_epoch_on_all_outer_training_samples",
        "missing_class_metric_policy": "all_dataset_classes_zero_division_0",
        "ovo_batch_scaling": "2 * training_partition_size * mean(weighted_pair_losses)",
        "selected_folds": args.folds, "complete_five_fold_run": bool(evaluated.all()),
        "evaluated_samples": int(evaluated.sum()), "total_samples": len(y),
        "combined_test_metrics": combined_metrics, "fold_results": fold_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_tag = "".join(map(str, args.folds))
    stem = f"{dataset_id}_{loss_name}_{criterion}_folds{fold_tag}_seed{args.seed}"
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / f"{stem}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "fold", "y_true", "y_pred"])
        for index in np.flatnonzero(evaluated):
            writer.writerow([index, fold_assignment[index], y[index], all_predictions[index]])
    coverage = "FULL 5-FOLD" if evaluated.all() else "PARTIAL RUN"
    print(f"[{coverage}] {dataset_id}/{loss_name}/{criterion}: "
          + " ".join(f"{key}={combined_metrics[key]:.4f}"
                     for key in ("accuracy", "macro_f1", "g_mean", "ba")), flush=True)
    print(f"Saved: {json_path}", flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["CAR"],
                        help="Dataset IDs, or ALL for all 17 datasets")
    parser.add_argument("--losses", nargs="+", choices=["ce", "ovr", "ovo"],
                        default=["ce", "ovr", "ovo"])
    parser.add_argument("--criteria", nargs="+", choices=["accuracy", "macro_f1"],
                        default=["accuracy", "macro_f1"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=PAPER_LEARNING_RATES)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Outer folds to run; a subset is only a partial experiment")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU threads; small tabular batches usually need only one")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "neural_network")
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("Epochs, patience, batch size and threads must be positive.")
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error("Learning rates must be finite and positive.")
    if len(set(args.folds)) != len(args.folds) or any(f < 1 or f > 5 for f in args.folds):
        parser.error("Folds must be distinct integers from 1 to 5.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu.")
    args.datasets = [dataset.upper() for dataset in args.datasets]
    if "ALL" in args.datasets:
        if len(args.datasets) != 1:
            parser.error("Use ALL alone, without other dataset IDs.")
        args.datasets = get_available_datasets(str(args.data_dir))
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    for dataset in args.datasets:
        for loss_name in args.losses:
            for criterion in args.criteria:
                run_experiment(dataset, loss_name, criterion, args)


if __name__ == "__main__":
    main()
