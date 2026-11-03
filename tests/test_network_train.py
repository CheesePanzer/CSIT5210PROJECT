"""公式、梯度、手写优化器和指标的校验；运行 python -m unittest discover -s tests。"""

import math
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import numpy as np
import torch

from src.network.network_train import (
    ManualAdam,
    PaperMLP,
    classification_loss,
    compute_metrics,
    predict,
    select_learning_rate,
    set_seed,
    train_one_epoch,
)


class LossTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [[0.2, -0.7, 1.1], [1.2, 0.4, -0.3], [-0.2, 0.8, 0.1]],
            dtype=torch.float64,
        )
        self.labels = torch.tensor([0, 1, 2])
        self.counts = torch.tensor([1, 1, 1])

    def test_ce_and_ovr_match_probability_definitions(self):
        probabilities = self.logits.exp()
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        expected_ce = -sum(math.log(probabilities[n, y])
                           for n, y in enumerate(self.labels)) / 3
        expected_ovr = -sum(
            math.log(probabilities[n, i] if i == y else 1 - probabilities[n, i])
            for n, y in enumerate(self.labels) for i in range(3)
        ) / 3
        for name, expected in (("ce", expected_ce), ("ovr", expected_ovr)):
            actual = classification_loss(self.logits, self.labels, name, self.counts)
            self.assertAlmostEqual(actual.item(), expected, places=12)

    def test_ovo_matches_ordered_pairs_in_paper(self):
        labels = torch.tensor([0, 0, 0, 1, 2])
        counts = torch.tensor([3, 1, 1])
        logits = self.logits[[0, 1, 2, 0, 1]]
        expected = 0.0
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for n, label in enumerate(labels.tolist()):
                    if label in (i, j):
                        other = j if label == i else i
                        expected += math.log1p(math.exp(float(logits[n, other] - logits[n, label]))) / int(counts[i] + counts[j])
        actual = classification_loss(logits, labels, "ovo", counts)
        self.assertAlmostEqual(actual.item(), expected, places=12)
        # 对完整训练分区按任意大小切 batch 后，加权均值仍等于论文目标。
        batched = sum(
            len(labels[start:end]) * classification_loss(
                logits[start:end], labels[start:end], "ovo", counts
            ).item() for start, end in ((0, 2), (2, 5))
        ) / len(labels)
        self.assertAlmostEqual(batched, expected, places=12)

    def test_gradients_and_extreme_logits(self):
        for name in ("ce", "ovr", "ovo"):
            logits = self.logits.clone().requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(
                lambda values: classification_loss(values, self.labels, name, self.counts),
                (logits,),
            ))
            extreme = torch.tensor([[1000.0, -1000.0, 0.0]], requires_grad=True)
            loss = classification_loss(extreme, torch.tensor([1]), name, self.counts)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(torch.isfinite(extreme.grad).all().item())


class TrainingTests(unittest.TestCase):
    def test_early_stopping_on_unchanged_validation_score(self):
        torch.set_num_threads(1)
        features = np.eye(3, dtype=np.float32)
        labels = np.arange(3)
        args = SimpleNamespace(
            device="cpu", learning_rates=[1e-12], max_epochs=8,
            patience=2, batch_size=3,
        )
        # 极小学习率使分类结果保持不变，验证 plateau 应触发早停。
        with redirect_stdout(StringIO()):
            best, trials = select_learning_rate(
                features, labels, features, labels, 3, "ce", "accuracy", args, 42
            )
        self.assertEqual(best["best_epoch"], 1)
        self.assertEqual(len(trials[0]["history"]), 3)

    def test_adam_matches_scalar_recurrence(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        optimizer = ManualAdam([parameter], lr=0.01)
        expected, first, second = 1.0, 0.0, 0.0
        for step, gradient in enumerate((2.0, -0.5, 0.25), start=1):
            optimizer.zero_grad()
            parameter.grad = torch.tensor([gradient], dtype=torch.float64)
            optimizer.step()
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient ** 2
            expected -= 0.01 * (first / (1 - 0.9 ** step)) / (
                math.sqrt(second / (1 - 0.999 ** step)) + 1e-8
            )
            self.assertAlmostEqual(parameter.item(), expected, places=12)

    def test_training_learns_with_all_three_losses(self):
        torch.set_num_threads(1)
        features = torch.eye(3).repeat(8, 1)
        labels = torch.arange(3).repeat(8)
        counts = torch.bincount(labels)
        for loss_name in ("ce", "ovr", "ovo"):
            set_seed(42)
            model = PaperMLP(3, 3)
            optimizer = ManualAdam(model.parameters(), lr=0.005)
            for _ in range(12):
                train_one_epoch(model, features, labels, optimizer, loss_name, counts, 8)
            np.testing.assert_array_equal(predict(model, features), labels.numpy())


class MetricTests(unittest.TestCase):
    def test_hand_calculated_confusion_matrix(self):
        metrics = compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2], 3)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)
        self.assertAlmostEqual(metrics["ba"], 2 / 3)
        self.assertAlmostEqual(metrics["g_mean"], 0.25 ** (1 / 3))

    def test_missing_class_retains_zero_recall(self):
        metrics = compute_metrics([0, 1], [0, 1], 3)
        self.assertEqual(metrics["g_mean"], 0.0)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["ba"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
