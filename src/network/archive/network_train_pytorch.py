"""论文神经网络的 PyTorch 版本：仅损失函数按论文公式自行实现。

运行：py -3.11 -m src.network.network_train_pytorch --datasets CAR
网络层/Adam/DataLoader 用 PyTorch，指标用 sklearn/scipy，数据用组员接口。
训练循环显式包含前向传播、计算损失、反向传播、更新参数及验证早停。

阅读顺序：main -> run_experiment -> fit_and_select_lr -> train_one_epoch。
与 skorch 版的区别：本文件自己组织 epoch/batch 和早停，损失也按公式实现；
skorch 版把训练交给库，并调用库的基础损失后再组合论文目标。
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import gmean
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 下面导入组员已经完成的数据接口，训练代码直接复用。
from src.dataset_registry import get_available_datasets
from src.datasets import load_dataset
from src.preprocessing import preprocess_partition
from src.split import get_inner_subsets, get_outer_folds
from src.network.reproduce_split import make_split_config, sample_counts


# 根据当前文件位置定位项目根目录，默认的数据/输出路径不依赖终端所在位置。
ROOT = Path(__file__).resolve().parents[2]
# 论文搜索的 8 个学习率；每个外层折都在自己的验证集上重新选一次。
LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]


def seed_everything(seed):
    """固定随机来源，便于复现实验；不保证不同硬件/库版本完全一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PaperMLP(nn.Module):
    """输入 [B,d]，输出 [B,k]；B 是批次样本数，k 是类别数。"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        # input_dim 是当前训练分区 one-hot 后的列数，不能写死为原始特征数。
        # Linear 学习特征组合；LayerNorm 归一化隐藏向量；ReLU 引入非线性；
        # Dropout 仅在训练时随机置零 10% 的隐藏元素，减轻过拟合。
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # model(x) 会调用 forward；每类一个分数，并不是 k 个独立网络。
        return self.network(x)  # 返回 logits，不额外加 sigmoid 或 softmax。


class PaperLoss(nn.Module):
    """手写 CE (式6)、OVR (式9)、OVO (式12)，不调用现成损失函数。

    logits: [B,k] 的原始分数；targets: [B]，标签已经编码为 0 到 k-1。
    三种损失使用同一种网络和 argmax 预测，区别只在训练目标。
    """

    def __init__(self, name, class_counts):
        super().__init__()
        if name not in ("ce", "ovr", "ovo"):
            raise ValueError(f"Unknown loss: {name}")
        self.name = name
        # 必须统计整个训练分区，不能只统计当前 batch！
        # buffer 随 loss_fn.to(device) 移动，但不作为可训练参数更新。
        self.register_buffer("class_counts", torch.as_tensor(class_counts, dtype=torch.float64))

    def forward(self, logits, targets):
        # gather 按每行的真实标签取出 f_y；log_z 是 log(sum_i exp(f_i))。
        # logsumexp 是 PyTorch 的数值稳定函数，等价于 softmax 的归一化项。
        true_logits = logits.gather(1, targets[:, None]).squeeze(1)
        log_z = torch.logsumexp(logits, dim=1)
        if self.name == "ce":
            # -log(exp(f_y) / sum_i exp(f_i))
            return (log_z - true_logits).mean()

        k = logits.shape[1]
        # [B,k] 的布尔掩码：真实类别位置为 True，用来排除“与自身比较”。
        is_target = torch.arange(k, device=logits.device)[None, :] == targets[:, None]
        if self.name == "ovr":
            # -log(p_y) - sum_(i != y) log(1-p_i)。
            # log(1-p_i) 在 log 空间计算，避免 softmax 舍入成 1 后 log(0)。
            exclude_self = torch.eye(k, dtype=torch.bool, device=logits.device)
            # 扩展为 [B,k,k]；中间一维代表要排除的类别 i。
            # 将 f_i 置为 -inf 后，logsumexp 只会汇总其余类别。
            expanded = logits[:, None, :].expand(-1, k, -1)
            log_others = torch.logsumexp(
                expanded.masked_fill(exclude_self[None, :, :], -torch.inf), dim=2
            )
            negative_terms = (log_z[:, None] - log_others).masked_fill(is_target, 0.0)
            # 每个样本：真实类正项 + 其他所有类负项；先对类别求和，再对样本平均。
            return (log_z - true_logits + negative_terms.sum(dim=1)).mean()

        # OVO：-log(p_(y,j)) = log(1 + exp(f_j - f_y))。
        pair_loss = torch.logaddexp(torch.zeros_like(logits), logits - true_logits[:, None])
        counts = self.class_counts.to(dtype=logits.dtype)
        # 对当前真实类 y 与每个类别 j，使用 n_y+n_j 作为分母。
        # 少数类之间的比较因分母较小而获得更大的权重。
        pair_size = counts[targets, None] + counts[None, :]
        if torch.any(pair_size <= 0):
            raise ValueError("Class counts must come from the current training partition.")
        weighted_loss = (pair_loss / pair_size).masked_fill(is_target, 0.0)
        # batch 均值乘 N_train，使其无偏估计论文完整数据集的式 (12)。
        return 2 * counts.sum() * weighted_loss.sum(dim=1).mean()


def make_loader(X, y, batch_size, shuffle=False):
    """把 NumPy 数据交给 PyTorch DataLoader，自动切 batch 和打乱训练顺序。"""
    # 特征用 float32，类别索引用 long；验证/测试必须保持原顺序以对齐标签。
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """完整遍历训练集一次；梯度计算和 Adam 内部更新仍由 PyTorch 完成。"""
    model.train()
    total_loss = 0.0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        # PyTorch 默认会累加梯度，每个新 batch 前必须清除上一个 batch 的梯度。
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)                 # 1. 前向传播
        loss = loss_fn(logits, targets)          # 2. 自定义论文损失
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss.")
        loss.backward()                         # 3. PyTorch 自动求导
        optimizer.step()                        # 4. PyTorch Adam 更新参数
        # 乘实际 batch 大小再累计，避免最后一个小 batch 被赋予过大的平均权重。
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """只做推理，返回按输入顺序排列的类别索引。"""
    model.eval()  # 关闭 Dropout；验证/测试不计算梯度。
    outputs = []
    for features, _ in loader:
        logits = model(features.to(device))
        # softmax 不改变分数大小顺序，因此直接对 logits 取 argmax 即可。
        # 神经网络 OVO 也按这个规则预测，不使用 SVM 的类别对投票。
        outputs.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def evaluate(y_true, y_pred, num_classes):
    """调用库计算四个论文指标；输出为 0~1 的小数，而非百分数。

    Accuracy：整体正确率；Macro-F1：各类 F1 的等权平均；
    G-mean：各类 recall 的几何平均（某类为 0 时结果为 0）；
    BA：各类 recall 的算术平均，避免多数类单独主导整体评价。
    """
    # 即使某个折缺少某类，也保留数据集全部 k 类；零分母按 0 处理。
    labels = list(range(num_classes))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "g_mean": float(gmean(recalls)),
        # BA 就是各类 recall 的宏平均。显式保留全部 k 类，包括缺失类别。
        "ba": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": recalls.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def fit_and_select_lr(X_train, y_train, X_val, y_val, k, loss_name, criterion, args, seed):
    """只用内层训练/验证数据，选择学习率及对应最佳轮数；不接触测试集。"""
    validation_loader = make_loader(X_val, y_val, args.batch_size)
    counts = np.bincount(y_train, minlength=k)
    trials = []
    for lr in args.learning_rates:
        seed_everything(seed)  # 每个学习率使用相同初始化。
        model = PaperMLP(X_train.shape[1], k).to(args.device)
        loss_fn = PaperLoss(loss_name, counts).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
        # waiting 记录连续未提升轮数；best_epoch 供之后全量重训使用。
        best_score, best_epoch, waiting = -math.inf, 0, 0
        history = []
        for epoch in range(1, args.max_epochs + 1):
            loss = train_one_epoch(model, train_loader, loss_fn, optimizer, args.device)
            predictions = predict(model, validation_loader, args.device)
            # criterion 是选参标准 accuracy 或 macro_f1，与训练损失不是一回事。
            score = evaluate(y_val, predictions, k)[criterion]
            history.append({"epoch": epoch, "train_loss": loss, "val_score": score})
            if score > best_score:
                # 只有严格提升才重置等待计数；相同分数保留较早的 epoch。
                best_score, best_epoch, waiting = score, epoch, 0
            else:
                waiting += 1
            if waiting >= args.patience:
                break
        trials.append({"lr": lr, "val_score": best_score, "best_epoch": best_epoch, "history": history})
        print(f"    lr={lr:g} val_{criterion}={best_score:.4f} "
              f"best_epoch={best_epoch} ran={len(history)}", flush=True)
    return max(trials, key=lambda trial: trial["val_score"]), trials


def run_experiment(dataset_id, loss_name, criterion, args):
    """运行一个“数据集 + 损失 + 选参标准”的实验，包含选定的外层测试折。"""
    # X 保留原始 DataFrame；y 是整数标签；meta 提供类别、类别列和官方划分。
    X, y, meta = load_dataset(dataset_id, data_dir=str(args.data_dir))
    outer_folds = get_outer_folds(X, y, metadata=meta, seed=args.seed)
    split_config = make_split_config(X, y, meta, args.data_dir, args.seed)
    k = meta["n_classes"]
    # 每个样本只在所属外层测试折中预测一次；-1 表示该样本尚未测试。
    all_predictions = np.full(len(y), -1, dtype=np.int64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    fold_results = []

    for fold in args.folds:
        # 外层留出约 1/5 数据测试；其余约 4/5 才能用于训练和验证。
        train_val_idx, test_idx = outer_folds[fold - 1]
        # 论文神经网络只使用内层五折中的第一个划分，以减少调参开销。
        train_rel, val_rel = get_inner_subsets(X, train_val_idx, y, seed=args.seed)[0]
        # 内层函数返回的是相对于 train_val_idx 的位置，先转换成全局行索引。
        train_idx, val_idx = train_val_idx[train_rel], train_val_idx[val_rel]
        # 预处理只在 train_idx 上拟合，验证集只 transform，避免信息泄漏。
        X_train, X_val = preprocess_partition(X, train_idx, val_idx, meta["categorical_cols"])
        print(f"[{dataset_id}/{loss_name}/{criterion}] fold={fold}/5 "
              f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
        best, trials = fit_and_select_lr(
            X_train, y[train_idx], X_val, y[val_idx], k,
            loss_name, criterion, args, args.seed + fold,
        )

        # 选好参数后重新拟合预处理器、新建模型，用整个外层训练集重训。
        X_refit, X_test = preprocess_partition(X, train_val_idx, test_idx, meta["categorical_cols"])
        seed_everything(args.seed + fold)
        model = PaperMLP(X_refit.shape[1], k).to(args.device)
        # 全量重训的样本数变化了，OVO 类别计数也必须重新计算。
        loss_fn = PaperLoss(loss_name, np.bincount(y[train_val_idx], minlength=k)).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"])
        train_loader = make_loader(X_refit, y[train_val_idx], args.batch_size, shuffle=True)
        # 论文未明确全量重训如何早停；这里复用内层最佳 epoch 数。
        # 测试集始终不参与学习率选择和训练轮数选择。
        for _ in range(best["best_epoch"]):
            train_one_epoch(model, train_loader, loss_fn, optimizer, args.device)
        predictions = predict(model, make_loader(X_test, y[test_idx], args.batch_size), args.device)
        # 写回样本的原始位置，才能在五折完成后合并计算总体指标。
        all_predictions[test_idx] = predictions
        fold_ids[test_idx] = fold
        metrics = evaluate(y[test_idx], predictions, k)
        fold_results.append({
            "fold": fold, "seed": args.seed + fold, "best_lr": best["lr"],
            "refit_epochs": best["best_epoch"], "trials": trials, "test_metrics": metrics,
            "inner_input_dim": X_train.shape[1], "refit_input_dim": X_refit.shape[1],
            # 结果只展示数量；完整索引可用 reproduce_split 按需重建并导出。
            "sample_counts": sample_counts(train_val_idx, test_idx, train_idx, val_idx),
        })
        print(f"    test accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)

    mask = all_predictions >= 0
    # 合并各折样本预测后计算指标，而不是对五个折的指标直接求平均。
    metrics = evaluate(y[mask], all_predictions[mask], k)
    result = {
        "implementation": "pytorch_components_custom_paper_losses",
        "dataset": dataset_id, "loss": loss_name, "criterion": criterion,
        "class_labels": meta["class_labels"], "seed": args.seed,
        "split_config": split_config,
        "learning_rates": args.learning_rates, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "optimizer": "torch.optim.Adam", "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
        "device": args.device, "torch_version": torch.__version__,
        "refit_policy": "inner_best_epoch_on_all_outer_training_samples",
        "missing_class_metric_policy": "all_dataset_classes_zero_division_0",
        "ovo_batch_scaling": "2 * N_train * batch_mean(weighted_pair_loss)",
        "selected_folds": args.folds, "complete_five_fold_run": bool(mask.all()),
        "evaluated_samples": int(mask.sum()), "total_samples": len(y),
        "combined_test_metrics": metrics, "fold_results": fold_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # JSON 保存参数、历史和指标；CSV 保存每个测试样本的真实/预测标签。
    # 文件名相同会覆盖，比较不同训练设置时可指定不同 --output-dir。
    stem = f"{dataset_id}_{loss_name}_{criterion}_folds{''.join(map(str, args.folds))}_seed{args.seed}"
    path = args.output_dir / f"{stem}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({
        "sample_index": np.flatnonzero(mask), "fold": fold_ids[mask],
        "y_true": y[mask], "y_pred": all_predictions[mask],
    }).to_csv(args.output_dir / f"{stem}_predictions.csv", index=False)
    status = "FULL 5-FOLD" if mask.all() else "PARTIAL RUN"
    print(f"[{status}] " + " ".join(f"{key}={metrics[key]:.4f}"
          for key in ("accuracy", "macro_f1", "g_mean", "ba")), flush=True)
    print(f"Saved: {path}", flush=True)
    return result


def parse_args():
    """读取终端参数；默认值对应论文设置，也允许用缩减参数快速检查代码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["CAR"], help="Dataset IDs or ALL")
    parser.add_argument("--losses", nargs="+", choices=["ce", "ovr", "ovo"], default=["ce", "ovr", "ovo"])
    parser.add_argument("--criteria", nargs="+", choices=["accuracy", "macro_f1"], default=["accuracy", "macro_f1"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=LEARNING_RATES)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "neural_network_pytorch")
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("Epochs, patience, batch size and threads must be positive.")
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error("Learning rates must be finite and positive.")
    if len(set(args.folds)) != len(args.folds) or any(f < 1 or f > 5 for f in args.folds):
        parser.error("Folds must be distinct integers from 1 to 5.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu.")
    args.datasets = [name.upper() for name in args.datasets]
    # ALL 代表扫描 data/raw 下的所有数据集，不是名为 ALL 的数据文件夹。
    if "ALL" in args.datasets:
        if len(args.datasets) != 1:
            parser.error("Use ALL alone.")
        args.datasets = get_available_datasets(str(args.data_dir))
    return args


def main():
    """实验入口：依次遍历数据集、三种损失和两种选参标准。"""
    args = parse_args()
    torch.set_num_threads(args.threads)
    for dataset_id in args.datasets:
        for loss_name in args.losses:
            for criterion in args.criteria:
                run_experiment(dataset_id, loss_name, criterion, args)


if __name__ == "__main__":
    # 被其他文件 import 时只加载定义，使用 python -m 运行时才开始实验。
    main()
