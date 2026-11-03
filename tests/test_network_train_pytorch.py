"""检查新版本自定义论文损失，以及与库组件的衔接。"""

import math
import unittest

import numpy as np
import torch

from src.network.network_train_pytorch import (
    PaperLoss, PaperMLP, evaluate, make_loader, predict, seed_everything, train_one_epoch,
)


class PyTorchVersionTests(unittest.TestCase):
    def test_ce_and_ovr_match_paper_probabilities(self):
        logits = torch.tensor([[0.1, 0.8, -0.5], [0.7, 0.3, -0.2]], dtype=torch.float64)
        labels = torch.tensor([1, 0])
        probabilities = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
        ce = -sum(math.log(probabilities[n, y]) for n, y in enumerate(labels)) / 2
        ovr = -sum(math.log(probabilities[n, i] if i == y else 1 - probabilities[n, i])
                   for n, y in enumerate(labels) for i in range(3)) / 2
        for name, expected in (("ce", ce), ("ovr", ovr)):
            self.assertAlmostEqual(PaperLoss(name, [1, 1, 0])(logits, labels).item(), expected, places=12)

    def test_ovo_ordered_pair_objective_and_batch_scaling(self):
        logits = torch.tensor([[0.2, 0.9, -0.4], [0.8, 0.3, -0.1], [-0.1, 0.2, 0.7]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1])
        counts = [2, 1, 0]
        # 从论文式 (11) 按有序类别对独立求和，包含训练中未出现的第三类。
        expected = 0.0
        for i in range(3):
            for j in range(3):
                if i == j or counts[i] + counts[j] == 0:
                    continue
                for n, y in enumerate(labels.tolist()):
                    if y in (i, j):
                        other = j if y == i else i
                        expected += math.log1p(math.exp(float(logits[n, other] - logits[n, y]))) / (counts[i] + counts[j])
        loss_fn = PaperLoss("ovo", counts)
        self.assertAlmostEqual(loss_fn(logits, labels).item(), expected, places=12)
        batched = (loss_fn(logits[:1], labels[:1]) + 2 * loss_fn(logits[1:], labels[1:])) / 3
        self.assertAlmostEqual(batched.item(), expected, places=12)

    def test_gradients_and_extreme_values(self):
        for name in ("ce", "ovr", "ovo"):
            loss_fn = PaperLoss(name, [1, 1, 1])
            logits = torch.tensor([[0.2, -0.7, 1.1]], dtype=torch.float64, requires_grad=True)
            labels = torch.tensor([1])
            self.assertTrue(torch.autograd.gradcheck(lambda x: loss_fn(x, labels), (logits,)))
            extreme = torch.tensor([[1000.0, -1000.0, 0.0]], requires_grad=True)
            loss = loss_fn(extreme, labels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(torch.isfinite(extreme.grad).all().item())

    def test_metrics_include_missing_classes(self):
        metrics = evaluate([0, 1], [0, 1], 3)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["ba"], 2 / 3)
        self.assertEqual(metrics["g_mean"], 0)

    def test_library_optimizer_and_loader_learn_with_each_loss(self):
        torch.set_num_threads(1)
        X = np.tile(np.eye(3, dtype=np.float32), (8, 1))
        y = np.tile(np.arange(3), 8)
        for name in ("ce", "ovr", "ovo"):
            seed_everything(42)
            model = PaperMLP(3, 3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            loader = make_loader(X, y, 8, shuffle=True)
            loss_fn = PaperLoss(name, np.bincount(y))
            for _ in range(12):
                train_one_epoch(model, loader, loss_fn, optimizer, "cpu")
            np.testing.assert_array_equal(predict(model, make_loader(X, y, 8), "cpu"), y)


if __name__ == "__main__":
    unittest.main()
