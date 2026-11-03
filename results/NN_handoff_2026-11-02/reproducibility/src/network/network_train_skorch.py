r"""调包版：skorch 训练/早停 + PyTorch 损失模块，只自行组合论文损失。

运行：.venv\Scripts\python.exe -m src.network.network_train_skorch --datasets CAR

阅读顺序：main -> run_experiment -> build_classifier -> classifier.fit/predict。
本文件只负责搭建网络、组合损失和组织实验配置；skorch 内部执行 epoch/batch、
反向传播、Adam 更新和早停，不需要另外编写 train_one_epoch。
"""

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import gmean
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, recall_score
# skorch 是 PyTorch 的高层训练接口：网络和基础损失仍来自 PyTorch。
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch import nn
from torch.nn import functional as F

# 复用组员的原始数据读取、预处理、官方/分层划分接口。
from src.dataset_registry import get_available_datasets
from src.datasets import load_dataset
from src.preprocessing import preprocess_partition
from src.split import get_inner_subsets, get_outer_folds
from src.network.reproduce_split import make_split_config, sample_counts


# 默认数据和输出路径以项目根目录为基准，不依赖当前终端目录。
ROOT = Path(__file__).resolve().parents[2]
# 论文使用的学习率候选；在每个外层折的内层验证集上独立选择。
LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]


class PaperMLP(nn.Module):
    """论文 MLP：输入 [B,d]，输出 [B,k] 的类别分数；B 是批次大小。"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        # d 是当前分区预处理后的维度；k 是整个数据集的类别数。
        # 两个隐藏层后均依次接归一化、非线性激活和训练时的随机丢弃。
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # 返回原始 logits，基础损失模块内部负责所需的概率计算。
        return self.layers(x)


class CombinedPaperLoss(nn.Module):
    """调用现成 CE/BCE 损失；这里只组合类别比较、权重和求和方式。

    logits: [B,k]；targets: [B]，整数标签为 0 到 k-1。
    网络结构和最终 argmax 预测相同，CE/OVR/OVO 只改变训练目标。
    """

    def __init__(self, name, class_counts):
        super().__init__()
        if name not in ("ce", "ovr", "ovo"):
            raise ValueError(f"Unknown loss: {name}")
        self.name = name
        # CE 接收原始 logits 和类别索引；内部已经包含 log_softmax。
        self.ce = nn.CrossEntropyLoss()
        # reduction='none' 返回每个二分类项，方便按论文对类别求和/加权。
        # 不能直接使用默认均值，否则会额外除以类别数，改变损失尺度。
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # 整个当前训练分区的类别计数，不是一个 batch 的计数。
        # buffer 随 criterion 移动到 CPU/GPU，但不会被 Adam 当成参数更新。
        self.register_buffer("counts", torch.as_tensor(class_counts, dtype=torch.float64))

    def forward(self, logits, targets):
        if self.name == "ce":
            return self.ce(logits, targets)

        k = logits.shape[1]
        # 类别索引转成 [B,k] 的 0/1 目标：真实类为 1，其余类为 0。
        one_hot = F.one_hot(targets, num_classes=k).to(dtype=logits.dtype)
        if self.name == "ovr":
            # 不能直接 BCEWithLogitsLoss(logits, one_hot)，那是独立 sigmoid。
            # 对每类 i 构造 f_i - logsumexp(f_j, j!=i) 作为二分类 logit：
            # sigmoid(该 logit) 恰好等于标准多分类 softmax 的 p_i。
            mask = torch.eye(k, dtype=torch.bool, device=logits.device)
            # 扩展成 [B,k,k] 后屏蔽对角线，只汇总类别 i 以外的分数。
            # logsumexp 和 BCEWithLogitsLoss 都是库函数，可稳定处理很大/很小的 logits。
            others = logits[:, None, :].expand(-1, k, -1).masked_fill(mask[None], -torch.inf)
            binary_logits = logits - torch.logsumexp(others, dim=-1)
            # 先对 k 个 OVR 二分类损失求和，再对 B 个样本平均，匹配论文式 (9)。
            return self.bce(binary_logits, one_hot).sum(dim=1).mean()

        # OVO 的二分类 logit 是 f_y-f_j，目标为 1，直接调用 BCEWithLogitsLoss。
        true_logits = logits.gather(1, targets[:, None])
        # 对每个样本，将真实类别 y 分别与其他类别 j 比较；目标总是“y 获胜”。
        pair_losses = self.bce(true_logits - logits, torch.ones_like(logits))
        counts = self.counts.to(dtype=logits.dtype)
        # 类别对权重为 1/(n_y+n_j)，不能用普通、无权重的 BCE 替代整个 OVO 目标。
        pair_sizes = counts[targets, None] + counts[None, :]
        if torch.any(pair_sizes <= 0):
            raise ValueError("Class counts must describe this training partition.")
        weighted = (pair_losses / pair_sizes).masked_fill(one_hot.bool(), 0.0)
        # 排除 y 与自身的比较；一个样本实际保留 k-1 个比较。
        # 保留论文有序类别对的系数 2；N 将 batch 均值还原到式 (12) 的尺度。
        return 2 * counts.sum() * weighted.sum(dim=1).mean()


def seed_everything(seed):
    """控制初始化和数据打乱的随机性，方便复现；不同平台仍可能有数值差异。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_classifier(input_dim, k, loss_name, counts, lr, args,
                     validation=None, criterion="macro_f1", epochs=None):
    """配置一个 skorch 分类器；validation=(X_val,y_val)，None 表示全量重训。"""
    callbacks = []
    if validation is not None:
        # criterion 在这里是选参标准，不是下面传给 skorch 的损失类。
        # make_scorer 将 sklearn 的 Macro-F1 包装为 skorch 能调用的评分接口。
        score = "accuracy" if criterion == "accuracy" else make_scorer(
            f1_score, average="macro", labels=list(range(k)), zero_division=0
        )
        callbacks = [
            # 每轮结束后在整个验证集评分；必须先记录 valid_score，早停才能读取。
            ("paper_score", EpochScoring(score, name="valid_score", lower_is_better=False)),
            # 分数越大越好；threshold=0 表示只认严格提升。
            # load_best=True 在训练结束时恢复验证分数最高轮的网络权重。
            ("paper_early_stop", EarlyStopping(
                monitor="valid_score", patience=args.patience, lower_is_better=False,
                threshold=0, threshold_mode="abs", load_best=True,
            )),
        ]
    return NeuralNetClassifier(
        # 双下划线参数会转发给对应组件的构造函数，例如 module__input_dim。
        module=PaperMLP, module__input_dim=input_dim, module__num_classes=k,
        criterion=CombinedPaperLoss, criterion__name=loss_name, criterion__class_counts=counts,
        optimizer=torch.optim.Adam, lr=lr, batch_size=args.batch_size,
        max_epochs=epochs if epochs is not None else args.max_epochs,
        # 显式使用我们划好的验证集，避免 skorch 又随机切一次数据。
        # 重训时设 None，表示用传入 fit 的全部数据，不留内部验证集。
        train_split=predefined_split(Dataset(*validation)) if validation is not None else None,
        callbacks=callbacks, classes=np.arange(k),
        # 这里的 softmax 只用于 predict/predict_proba，不改变损失函数收到的 logits。
        # softmax 后取 argmax 等价于直接对 logits 取 argmax，OVO 也不需要投票。
        predict_nonlinearity=partial(F.softmax, dim=-1),
        iterator_train__shuffle=True, device=args.device, verbose=0,
    )


def evaluate(y_true, predictions, k):
    """调用 sklearn/scipy 计算论文指标，返回 0~1 小数及混淆矩阵。

    Accuracy 是整体正确率，Macro-F1 对各类别的 F1 等权平均；
    G-mean 是各类 recall 的几何平均，BA 是各类 recall 的算术平均。
    后三者按类别评价，更能反映少数类的表现。
    """
    # 显式保留全部 k 类；个别折未出现某类时，该类 recall/F1 的零分母记为 0。
    labels = list(range(k))
    recalls = recall_score(y_true, predictions, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "g_mean": float(gmean(recalls)),
        # BA 是各类别 recall 的宏平均；这里保留固定的类别集合。
        "ba": float(recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
    }


def run_experiment(dataset_id, loss_name, criterion, args):
    """完成一组配置的外层测试、内层选参、全量重训和结果保存。"""
    # X 是原始 DataFrame；meta 含类别列和可用的 KEEL 官方五折划分。
    X, y, meta = load_dataset(dataset_id, data_dir=str(args.data_dir))
    y = np.asarray(y, dtype=np.int64)
    k = meta["n_classes"]
    outer_folds = get_outer_folds(X, y, metadata=meta, seed=args.seed)
    split_config = make_split_config(X, y, meta, args.data_dir, args.seed)
    # -1 表示尚未测试；预测值按原始行索引写回，便于合并五折结果。
    predictions = np.full(len(y), -1, dtype=np.int64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    folds = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        # 外层约 1/5 留作测试，其余数据用于训练与验证。
        train_val_idx, test_idx = outer_folds[fold - 1]
        # 论文神经网络仅使用内层五折的第一个划分。
        train_rel, val_rel = get_inner_subsets(X, train_val_idx, y, seed=args.seed)[0]
        train_idx, val_idx = train_val_idx[train_rel], train_val_idx[val_rel]
        # train_rel/val_rel 是相对位置，上一行已转为全局索引。
        # 预处理在内层训练集拟合，只对验证集 transform，防止验证信息泄漏。
        X_train, X_val = preprocess_partition(X, train_idx, val_idx, meta["categorical_cols"])
        print(f"[{dataset_id}/{loss_name}/{criterion}] fold {fold}/5", flush=True)
        trials = []
        # 这里只循环实验配置；epoch/batch/反向传播/Adam/早停全部由 skorch 执行。
        for lr in args.learning_rates:
            # 各候选学习率从相同随机状态出发，避免初始化差异干扰比较。
            seed_everything(args.seed + fold)
            classifier = build_classifier(
                X_train.shape[1], k, loss_name, np.bincount(y[train_idx], minlength=k), lr, args,
                validation=(X_val, y[val_idx]), criterion=criterion,
            )
            classifier.fit(X_train, y[train_idx])
            # skorch 自动记录每轮历史；取最高分的第一轮，同分保留较早 epoch。
            scores = classifier.history[:, "valid_score"]
            best_index = int(np.argmax(scores))
            history = [{"epoch": int(row["epoch"]), "train_loss": float(row["train_loss"]),
                        "val_score": float(row["valid_score"])} for row in classifier.history]
            trial = {"lr": lr, "score": float(scores[best_index]),
                     "epochs": int(classifier.history[best_index, "epoch"]), "history": history}
            trials.append(trial)
            print(f"    lr={lr:g} val={trial['score']:.4f} best_epoch={trial['epochs']}", flush=True)

        best = max(trials, key=lambda trial: trial["score"])
        # 参数选择完成；用整个外层训练分区重新拟合预处理器。
        # one-hot 列数可能变化，所以重训必须新建网络，并重新统计各类样本数。
        X_refit, X_test = preprocess_partition(X, train_val_idx, test_idx, meta["categorical_cols"])
        seed_everything(args.seed + fold)
        classifier = build_classifier(
            X_refit.shape[1], k, loss_name, np.bincount(y[train_val_idx], minlength=k),
            best["lr"], args, epochs=best["epochs"],
        )
        # 论文没有明确全量重训时如何早停：这里复用内层选定的 epoch 数。
        # 未提供 validation，不会再用测试集做早停或选择学习率。
        classifier.fit(X_refit, y[train_val_idx])
        predictions[test_idx] = classifier.predict(X_test)
        fold_ids[test_idx] = fold
        folds.append({
            "fold": fold, "best_lr": best["lr"], "refit_epochs": best["epochs"], "trials": trials,
            "inner_input_dim": X_train.shape[1], "refit_input_dim": X_refit.shape[1],
            # 不再把样本索引长列表写入报告，按需用复现工具导出。
            "sample_counts": sample_counts(train_val_idx, test_idx, train_idx, val_idx),
            "test_metrics": evaluate(y[test_idx], predictions[test_idx], k),
        })

    covered = predictions >= 0
    # 合并所有已测试样本后计算最终指标，不能直接平均各折的 Macro-F1/G-mean。
    # 只选择部分外层折时，下面的结果只代表部分实验。
    result = {
        "implementation": "skorch_with_composed_torch_losses",
        "dataset": dataset_id, "loss": loss_name, "criterion": criterion, "seed": args.seed,
        "split_config": split_config,
        "class_labels": meta["class_labels"], "selected_folds": args.folds,
        "max_epochs": args.max_epochs, "batch_size": args.batch_size, "patience": args.patience,
        "learning_rates": args.learning_rates, "optimizer": "torch.optim.Adam",
        "adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "device": args.device,
        "refit_policy": "inner_best_epoch_on_all_outer_training_samples",
        "missing_class_metric_policy": "all_dataset_classes_zero_division_0",
        "ovo_batch_scaling": "2 * N_train * batch_mean(weighted_pair_loss)",
        "complete_five_fold_run": bool(covered.all()), "evaluated_samples": int(covered.sum()),
        "total_samples": len(y), "combined_test_metrics": evaluate(y[covered], predictions[covered], k),
        "fold_results": folds,
    }
    stem = f"{dataset_id}_{loss_name}_{criterion}_folds{''.join(map(str, args.folds))}_seed{args.seed}"
    # JSON 保存参数、划分数量/复现设置、各轮记录与指标；CSV 保存逐样本预测。
    # 同名配置再次运行会覆盖，可通过 --output-dir 指定不同目录。
    path = args.output_dir / f"{stem}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_index": np.flatnonzero(covered), "fold": fold_ids[covered],
                  "y_true": y[covered], "y_pred": predictions[covered]}).to_csv(
        args.output_dir / f"{stem}_predictions.csv", index=False
    )
    status = "FULL 5-FOLD" if covered.all() else "PARTIAL RUN"
    print(f"[{status}] {result['combined_test_metrics']}", flush=True)
    print(f"Saved: {path}", flush=True)
    return result


def parse_args():
    """终端参数：默认是论文训练设置，也可缩减学习率/轮数/折数快速试跑。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["CAR"])
    parser.add_argument("--losses", nargs="+", choices=["ce", "ovr", "ovo"], default=["ce", "ovr", "ovo"])
    parser.add_argument("--criteria", nargs="+", choices=["accuracy", "macro_f1"], default=["accuracy", "macro_f1"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=LEARNING_RATES)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "neural_network_skorch")
    args = parser.parse_args()
    if min(args.max_epochs, args.batch_size, args.patience, args.threads) < 1:
        parser.error("Epochs, batch size, patience and threads must be positive.")
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error("Learning rates must be finite and positive.")
    if len(set(args.folds)) != len(args.folds) or any(f not in range(1, 6) for f in args.folds):
        parser.error("Folds must be distinct integers from 1 to 5.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use cpu.")
    args.datasets = [name.upper() for name in args.datasets]
    # ALL 会扫描现有数据集目录，不需要在代码里逐个写数据集名。
    if "ALL" in args.datasets:
        if len(args.datasets) != 1:
            parser.error("Use ALL alone.")
        args.datasets = get_available_datasets(str(args.data_dir))
    return args


def main():
    """依次运行数据集、损失和选参标准的配置组合；训练细节由 skorch 执行。"""
    args = parse_args()
    torch.set_num_threads(args.threads)
    for dataset in args.datasets:
        for loss in args.losses:
            for criterion in args.criteria:
                run_experiment(dataset, loss, criterion, args)


if __name__ == "__main__":
    # import 时不开始训练；通过 python -m 运行本模块时才进入 main。
    main()
