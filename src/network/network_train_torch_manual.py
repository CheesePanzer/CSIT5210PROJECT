r"""PyTorch 手动训练版 v2：损失按论文的 mini-batch 实现组合。

网络层/Adam/DataLoader 用 PyTorch，指标用 sklearn/scipy，数据用组员接口。
训练循环显式包含前向传播、计算损失、反向传播、更新参数及验证早停。
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 下面导入组员已经完成的数据接口，训练代码直接复用。
from src.dataset_registry import get_available_datasets
from src.datasets import load_dataset
from src.preprocessing import preprocess_partition
from src.split import get_inner_subsets, get_outer_folds
from src.network.split_metadata import make_split_config, sample_counts
from src.network.paper_losses import PaperLoss
from src.network.evaluation import PaperEvaluation


ROOT = Path(__file__).resolve().parents[2]
NETWORK_DIR = Path(__file__).resolve().parent
# 论文搜索的 8 个学习率；每个外层折都在自己的验证集上重新选一次。
LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
LOSSES = ["ce", "ovr", "ovo"]
CRITERIA = ["accuracy", "macro_f1"]
OUTER_FOLDS = [1, 2, 3, 4, 5]
MAX_EPOCHS = 40
PATIENCE = 10
BATCH_SIZE = 8
SEED = 42
THREADS = 1


def seed_everything(seed):
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


def make_loader(X, y, batch_size, shuffle=False):
    """把 NumPy 数据交给 PyTorch DataLoader，自动切 batch 和打乱训练顺序。"""
    # 特征用 float32，类别索引用 long；验证/测试必须保持原顺序以对齐标签。
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """完整遍历训练集一次"""
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


def fit_and_select_lr(X_train, y_train, X_val, y_val, k, loss_name, criterion, args, seed):
    """只用内层训练/验证数据，选择学习率及对应最佳轮数；不接触测试集。"""
    validation_loader = make_loader(X_val, y_val, BATCH_SIZE)
    evaluator = PaperEvaluation(k)
    trials = []
    for lr in LEARNING_RATES:
        seed_everything(seed)  # 每个学习率使用相同初始化。
        model = PaperMLP(X_train.shape[1], k).to(args.device)
        loss_fn = PaperLoss(
            loss_name,
            k,
            class_counts=np.bincount(y_train, minlength=k),
        ).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
        # waiting 记录连续未提升轮数；best_epoch 供之后全量重训使用。
        best_score, best_epoch, waiting = -math.inf, 0, 0
        history = []
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, loss_fn, optimizer, args.device)
            predictions = predict(model, validation_loader, args.device)
            # criterion 是选参标准 accuracy 或 macro_f1，与训练损失不是一回事。
            score = evaluator.score(y_val, predictions, criterion)
            history.append({"epoch": epoch, "train_loss": loss, "val_score": score})
            if score > best_score:
                # 只有严格提升才重置等待计数；相同分数保留较早的 epoch。
                best_score, best_epoch, waiting = score, epoch, 0
            else:
                waiting += 1
            if waiting >= PATIENCE:
                break
        trials.append({"lr": lr, "val_score": best_score, "best_epoch": best_epoch, "history": history})
        print(f"    lr={lr:g} val_{criterion}={best_score:.4f} "
              f"best_epoch={best_epoch} ran={len(history)}", flush=True)
    return max(trials, key=lambda trial: trial["val_score"]), trials


def run_experiment(dataset_id, loss_name, criterion, args):
    """运行一个“数据集 + 损失 + 选参标准”的实验，包含选定的外层测试折。"""
    # X 保留原始 DataFrame；y 是整数标签；meta 提供类别、类别列和官方划分。
    X, y, meta = load_dataset(dataset_id, data_dir=str(args.data_dir))
    outer_folds = get_outer_folds(X, y, metadata=meta, seed=SEED)
    split_config = make_split_config(X, y, meta, args.data_dir, SEED)
    k = meta["n_classes"]
    # 每个样本只在所属外层测试折中预测一次；-1 表示该样本尚未测试。
    all_predictions = np.full(len(y), -1, dtype=np.int64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    fold_results = []

    for fold in OUTER_FOLDS:
        # 外层留出约 1/5 数据测试；其余约 4/5 才能用于训练和验证。
        train_val_idx, test_idx = outer_folds[fold - 1]
        # 论文神经网络只使用内层五折中的第一个划分，以减少调参开销。
        train_rel, val_rel = get_inner_subsets(X, train_val_idx, y, seed=SEED)[0]
        # 内层函数返回的是相对于 train_val_idx 的位置，先转换成全局行索引。
        train_idx, val_idx = train_val_idx[train_rel], train_val_idx[val_rel]
        # 预处理只在 train_idx 上拟合，验证集只 transform，避免信息泄漏。
        X_train, X_val = preprocess_partition(X, train_idx, val_idx, meta["categorical_cols"])
        print(f"[{dataset_id}/{loss_name}/{criterion}] fold={fold}/5 "
              f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
        best, trials = fit_and_select_lr(
            X_train, y[train_idx], X_val, y[val_idx], k,
            loss_name, criterion, args, SEED + fold,
        )

        # 选好参数后重新拟合预处理器、新建模型，用整个外层训练集重训。
        X_refit, X_test = preprocess_partition(X, train_val_idx, test_idx, meta["categorical_cols"])
        seed_everything(SEED + fold)
        model = PaperMLP(X_refit.shape[1], k).to(args.device)
        # 提供class counts 以应对oi或者oj的计数
        loss_fn = PaperLoss(
            loss_name,
            k,
            class_counts=np.bincount(y[train_val_idx], minlength=k),
        ).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"])
        train_loader = make_loader(X_refit, y[train_val_idx], BATCH_SIZE, shuffle=True)
        # 论文未明确全量重训如何早停；这里复用内层最佳 epoch 数。
        # 测试集始终不参与学习率选择和训练轮数选择。
        for _ in range(best["best_epoch"]):
            train_one_epoch(model, train_loader, loss_fn, optimizer, args.device)
        predictions = predict(model, make_loader(X_test, y[test_idx], BATCH_SIZE), args.device)
        # 写回样本的原始位置，才能在五折完成后合并计算总体指标。
        all_predictions[test_idx] = predictions
        fold_ids[test_idx] = fold
        metrics = evaluate(y[test_idx], predictions, k)
        fold_results.append({
            "fold": fold, "seed": SEED + fold, "best_lr": best["lr"],
            "refit_epochs": best["best_epoch"], "trials": trials, "test_metrics": metrics,
            "inner_input_dim": X_train.shape[1], "refit_input_dim": X_refit.shape[1],
            # 结果只展示数量；完整索引可用 reproduce_split 按需重建并导出。
            "sample_counts": sample_counts(train_val_idx, test_idx, train_idx, val_idx),
        })
        print(f"    test accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)

    mask = all_predictions >= 0
    # 合并各折样本预测后计算指标，而不是对五个折的指标直接求平均。
    metrics = PaperEvaluation(k).evaluate(y[mask], all_predictions[mask])
    result = {
        "dataset": dataset_id, 
        "loss": loss_name, 
        "criterion": criterion,
        "class_labels": meta["class_labels"], 
        "seed": SEED,
        "split_config": split_config,
        "learning_rates": LEARNING_RATES,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "optimizer": "torch.optim.Adam",
        "device": args.device, 
        "torch_version": torch.__version__,
        "selected_folds": OUTER_FOLDS,
        "complete_five_fold_run": bool(mask.all()),
        "evaluated_samples": int(mask.sum()), 
        "total_samples": len(y),
        "combined_test_metrics": metrics, 
        "fold_results": fold_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # JSON 保存参数、历史和指标；CSV 保存每个测试样本的真实/预测标签。
    # 文件名相同会覆盖，比较不同训练设置时可指定不同 --output-dir。
    stem = f"{dataset_id}_{loss_name}_{criterion}_folds{''.join(map(str, OUTER_FOLDS))}_seed{SEED}"
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
    """Parse only settings that are expected to change between runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["CAR"], help="Dataset IDs or ALL")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=NETWORK_DIR / "results" / "neural_network_torch_manual_v2")
    args = parser.parse_args()
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
    torch.set_num_threads(THREADS)
    for dataset_id in args.datasets:
        for loss_name in LOSSES:
            for criterion in CRITERIA:
                run_experiment(dataset_id, loss_name, criterion, args)


if __name__ == "__main__":
    main()
