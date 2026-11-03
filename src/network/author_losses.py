"""对齐作者公开代码的实际 mini-batch 损失，而非全量公式的无偏缩放版本。

来源：https://www.csie.ntu.edu.tw/~cjlin/papers/ovo-ovr/ovo-ovr-code.zip
文件：multi_class_nn/ovo_loss.py。没有执行下载的代码；这里只重写相同计算。
"""

import torch
from torch import nn
from torch.nn import functional as F


class AuthorPaperLoss(nn.Module):
    """CE 用标准交叉熵，OVR/OVO 的 reduction 和计数范围与作者代码一致。"""

    def __init__(self, name, num_classes):
        super().__init__()
        if name not in ("ce", "ovr", "ovo"):
            raise ValueError(f"Unknown loss: {name}")
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")
        self.name = name
        self.num_classes = num_classes
        self.num_pairs = num_classes * (num_classes - 1) // 2

    def forward(self, logits, targets):
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError("logits must have shape [batch_size, num_classes].")
        if targets.ndim != 1 or len(targets) != len(logits) or len(targets) == 0:
            raise ValueError("targets must be a nonempty vector matching the batch size.")

        if self.name == "ce":
            return F.cross_entropy(logits, targets)

        is_target = F.one_hot(targets, num_classes=self.num_classes).bool()
        if self.name == "ovr":
            # 作者用 BCELoss() 默认 mean：对 B*k 项平均，而非仅对 B 个样本平均。
            # 概率必须来自多分类 softmax，不是各类独立 sigmoid。
            probabilities = F.softmax(logits, dim=1)
            return F.binary_cross_entropy(probabilities, is_target.to(logits.dtype))

        # 关键：n_i、n_j 来自本次 forward 的 batch，不来自完整训练集。
        counts = torch.bincount(targets, minlength=self.num_classes).to(logits.dtype)
        pair_sizes = counts[targets, None] + counts[None, :]
        # 每条样本的真实类必然在当前 batch 中，因此所有使用的分母均为正。
        true_logits = logits.gather(1, targets[:, None])
        pair_losses = -F.logsigmoid(true_logits - logits)
        weighted = (pair_losses / pair_sizes).masked_fill(is_target, 0.0)
        # 每个无序类别对的损失按该对在 batch 中的样本数平均，再除以总类别对数。
        # 即使某对在 batch 中完全缺席，分母仍是固定的 C(k,2)，不能只数出现的对。
        # 这里没有额外的 2、N_train 或 batch mean。
        return weighted.sum() / self.num_pairs
